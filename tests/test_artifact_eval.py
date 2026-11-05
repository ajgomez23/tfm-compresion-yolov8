"""Tests de la evaluación de calidad de artefactos exportados."""

from __future__ import annotations

import ast

import pytest

from tfm_edge.export.staging import (
    ArtifactStagingError,
    assert_graph_unchanged,
    build_ultralytics_metadata,
    metadata_as_onnx_strings,
    stage_artifact,
    stage_onnx_artifact,
    stage_openvino_artifact,
)
from tfm_edge.models.detector import build_validator_args

CLASES = ["person", "helmet", "gloves"]


def _onnx_minimo(ruta, metadatos=None):
    """Escribe un ONNX válido y diminuto, opcionalmente con metadatos previos."""
    import onnx
    from onnx import TensorProto, helper

    nodo = helper.make_node("Identity", ["entrada"], ["salida"])
    grafo = helper.make_graph(
        [nodo],
        "minimo",
        [helper.make_tensor_value_info("entrada", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("salida", TensorProto.FLOAT, [1, 3, 8, 8])],
    )
    modelo = helper.make_model(grafo, opset_imports=[helper.make_opsetid("", 17)])
    for clave, valor in (metadatos or {}).items():
        entrada = modelo.metadata_props.add()
        entrada.key, entrada.value = clave, valor
    onnx.save(modelo, str(ruta))
    return ruta


class TestMetadatosInyectados:
    """El mapeo de clases del artefacto es el del repositorio."""

    def test_los_nombres_van_indexados_por_entero(self) -> None:
        metadatos = build_ultralytics_metadata(CLASES)
        assert metadatos["names"] == {0: "person", 1: "helmet", 2: "gloves"}

        reconstruido = ast.literal_eval(metadata_as_onnx_strings(metadatos)["names"])
        assert reconstruido == metadatos["names"]
        assert all(isinstance(clave, int) for clave in reconstruido)

    def test_la_serializacion_a_onnx_es_reversible(self) -> None:
        """Ultralytics deshace esta conversión al leer; tiene que cuadrar."""
        metadatos = build_ultralytics_metadata(CLASES, input_resolution=512, batch=1)
        cadenas = metadata_as_onnx_strings(metadatos)
        assert all(isinstance(valor, str) for valor in cadenas.values())

        deshecho = {
            "stride": int(cadenas["stride"]),
            "task": cadenas["task"],
            "batch": int(cadenas["batch"]),
            "channels": int(cadenas["channels"]),
            "imgsz": ast.literal_eval(cadenas["imgsz"]),
            "names": ast.literal_eval(cadenas["names"]),
        }
        assert deshecho == metadatos

    def test_sin_clases_no_se_prepara_nada(self) -> None:
        """Preparar sin mapeo dejaría el vocabulario inventado de AutoBackend."""
        with pytest.raises(ArtifactStagingError, match="nombres de clase"):
            build_ultralytics_metadata([])


class TestElArtefactoMedidoNoSeToca:
    """La copia preparada tiene que ser el mismo modelo, y demostrarlo."""

    def test_el_original_conserva_sus_bytes(self, tmp_path) -> None:
        """De ello depende que 'artifact_size_bytes' siga siendo cierto."""
        original = _onnx_minimo(tmp_path / "m.onnx")
        antes = original.read_bytes()
        stage_onnx_artifact(original, tmp_path / "staged", CLASES)
        assert original.read_bytes() == antes

    def test_el_grafo_preparado_es_el_mismo(self, tmp_path) -> None:
        """La preparación solo puede añadir metadatos."""
        original = _onnx_minimo(tmp_path / "m.onnx")
        preparado = stage_onnx_artifact(original, tmp_path / "staged", CLASES)
        assert assert_graph_unchanged(original, preparado)

    def test_un_grafo_distinto_se_caza(self, tmp_path) -> None:
        """Si alguien sustituyera la preparación por una que reescribe pesos."""
        import onnx
        from onnx import TensorProto, helper

        uno = _onnx_minimo(tmp_path / "uno.onnx")
        grafo = helper.make_graph(
            [helper.make_node("Identity", ["entrada"], ["salida"])],
            "minimo",
            [helper.make_tensor_value_info("entrada", TensorProto.FLOAT, [1, 3, 16, 16])],
            [helper.make_tensor_value_info("salida", TensorProto.FLOAT, [1, 3, 16, 16])],
        )
        otro = tmp_path / "otro.onnx"
        onnx.save(helper.make_model(grafo, opset_imports=[helper.make_opsetid("", 17)]), str(otro))

        with pytest.raises(ArtifactStagingError, match="grafo distinto"):
            assert_graph_unchanged(uno, otro)

    def test_los_metadatos_previos_no_sobreviven(self, tmp_path) -> None:
        """Es lo que impedía cargar el INT8 de ONNX Runtime."""
        import onnx

        original = _onnx_minimo(
            tmp_path / "m.onnx",
            metadatos={"quantization": "onnxruntime", "names": "{0: 'otra_cosa'}"},
        )
        preparado = stage_onnx_artifact(original, tmp_path / "staged", CLASES)
        claves = {p.key: p.value for p in onnx.load(str(preparado)).metadata_props}
        assert "quantization" not in claves
        assert ast.literal_eval(claves["names"]) == {0: "person", 1: "helmet", 2: "gloves"}


class TestPreparacionDeOpenVino:
    """Ultralytics reconoce este formato por el nombre del directorio."""

    def _ir_falso(self, directorio, stem="modelo"):
        """Crea un par .xml/.bin que basta para la preparación."""
        directorio.mkdir(parents=True, exist_ok=True)
        (directorio / f"{stem}.xml").write_text("<net/>", encoding="utf-8")
        (directorio / f"{stem}.bin").write_bytes(b"\x00")
        return directorio

    def test_el_directorio_acaba_en_openvino_model(self, tmp_path) -> None:
        """Sin ese sufijo el artefacto ni se reconoce como formato admitido."""
        origen = self._ir_falso(tmp_path / "cualquier_nombre")
        destino = stage_openvino_artifact(origen, tmp_path / "staged", CLASES)
        assert destino.name.endswith("_openvino_model")

    def test_el_batch_del_yaml_va_como_entero(self, tmp_path) -> None:
        import yaml

        origen = self._ir_falso(tmp_path / "ir")
        destino = stage_openvino_artifact(origen, tmp_path / "staged", CLASES)
        metadatos = yaml.safe_load((destino / "metadata.yaml").read_text(encoding="utf-8"))
        assert isinstance(metadatos["batch"], int)
        assert metadatos["names"] == {0: "person", 1: "helmet", 2: "gloves"}

    def test_un_ir_sin_pesos_es_un_error(self, tmp_path) -> None:
        directorio = tmp_path / "ir"
        directorio.mkdir()
        (directorio / "modelo.xml").write_text("<net/>", encoding="utf-8")
        with pytest.raises(ArtifactStagingError, match="pesos"):
            stage_openvino_artifact(directorio, tmp_path / "staged", CLASES)

    def test_un_runtime_sin_preparacion_no_se_inventa(self, tmp_path) -> None:
        with pytest.raises(ArtifactStagingError, match="tensorrt"):
            stage_artifact(tmp_path / "x.plan", "tensorrt", tmp_path, CLASES)


class TestMismosArgumentosQueElCheckpoint:
    """La comparación solo vale si el protocolo de evaluación es el mismo."""

    def _base(self, **extra):
        argumentos = {
            "data_config": "d.yaml",
            "split": "val",
            "batch_size": 1,
            "input_resolution": 640,
            "conf_threshold": 0.001,
            "iou_threshold": 0.7,
        }
        argumentos.update(extra)
        return build_validator_args(**argumentos)

    @pytest.mark.parametrize(
        ("clave", "valor"),
        [
            # 'rect' agrupa por relación de aspecto y cambia el relleno, del que depende el criterio
            # de tamaño primario.
            ("rect", False),
            # Una referencia que pasa por FP16 subestima la caída del INT8.
            ("half", False),
        ],
    )
    def test_los_ajustes_que_invalidarian_la_comparacion(self, clave, valor) -> None:
        assert self._base()[clave] is valor

    def test_el_lote_no_altera_el_resto_del_protocolo(self) -> None:
        """El artefacto es de lote 1 y el checkpoint se evalúa con 16."""
        artefacto = self._base(batch_size=1)
        checkpoint = self._base(batch_size=16)
        diferencias = {
            clave
            for clave in set(artefacto) | set(checkpoint)
            if artefacto.get(clave) != checkpoint.get(clave)
        }
        assert diferencias == {"batch"}

    def test_save_json_es_lo_que_alimenta_las_detecciones(self) -> None:
        """Sin él el validador calcula métricas pero no acumula jdict."""
        assert "save_json" not in self._base()
        assert self._base(save_json=True)["save_json"] is True


class TestDondeVanLasDeteccionesCrudas:
    """El nombre del directorio del validador es lo que separa dos medidas."""

    @staticmethod
    def _script():
        """Carga el script del paso 11 como módulo."""
        import importlib.util
        from pathlib import Path

        ruta = Path(__file__).resolve().parent.parent / "scripts" / "11_evaluate_artifact.py"
        especificacion = importlib.util.spec_from_file_location("evaluate_artifact_script", ruta)
        assert especificacion is not None and especificacion.loader is not None
        modulo = importlib.util.module_from_spec(especificacion)
        especificacion.loader.exec_module(modulo)
        return modulo

    def _dir(
        self, experiment_id, runtime="onnxruntime", precision="int8", smoke_test=False, split="val"
    ):
        """Compone el directorio con los valores indicados."""
        from pathlib import Path

        return self._script().validator_dir_for(
            Path("salida"), experiment_id, runtime, precision, smoke_test, split
        )

    def test_dos_experimentos_no_comparten_destino(self) -> None:
        ptq = self._dir("s_ptq_int8-noexcl_5bcd71f0")
        qat = self._dir("s_qat_int8-noexcl_d4d403fa")
        assert ptq != qat, "el INT8 de QAT pisaría las detecciones del INT8 de PTQ"

    def test_la_prueba_de_humo_no_pisa_a_la_evaluacion_completa(self) -> None:
        """Cuarenta imágenes no pueden acabar donde estaban 1.213."""
        assert self._dir("s_ptq_int8-noexcl_5bcd71f0", smoke_test=True) != self._dir(
            "s_ptq_int8-noexcl_5bcd71f0"
        )

    def test_el_runtime_y_la_precision_siguen_separando(self) -> None:
        """Lo que ya funcionaba tiene que seguir funcionando."""
        base = self._dir("s_ptq_int8-noexcl_5bcd71f0")
        assert base != self._dir("s_ptq_int8-noexcl_5bcd71f0", runtime="openvino")
        assert base != self._dir("s_ptq_int8-noexcl_5bcd71f0", precision="fp32")

    def test_test_no_pisa_las_detecciones_de_val(self) -> None:
        assert self._dir("s_ptq_int8-noexcl_5bcd71f0", split="test") != self._dir(
            "s_ptq_int8-noexcl_5bcd71f0", split="val"
        )

    def test_ningun_directorio_nuevo_coincide_con_uno_antiguo(self) -> None:
        antiguo = "salida/validator_s_ptq_int8-noexcl_5bcd71f0_onnxruntime_int8"
        for split in ("val", "test"):
            for humo in (False, True):
                nuevo = self._dir("s_ptq_int8-noexcl_5bcd71f0", smoke_test=humo, split=split)
                assert nuevo.as_posix() != antiguo


class TestNombreDelVolcado:
    """El volcado lleva partición y dispositivo en el nombre."""

    _script = staticmethod(TestDondeVanLasDeteccionesCrudas._script)

    def _ruta(self, split="val", device_id="workstation_cpu", smoke_test=False):
        """Compone la ruta del volcado con los valores indicados."""
        from pathlib import Path

        return self._script().dump_path_for(
            Path("salida"),
            "s_ptq_int8-noexcl_5bcd71f0",
            "onnxruntime",
            "int8",
            split,
            640,
            device_id,
            smoke_test,
        )

    def test_val_y_test_no_comparten_volcado(self) -> None:
        """La partición separa los dos volcados del mismo artefacto."""
        assert self._ruta(split="val") != self._ruta(split="test")

    def test_dos_dispositivos_no_comparten_volcado(self) -> None:
        assert self._ruta(device_id="workstation_cpu") != self._ruta(device_id="edge_cpu_arm")

    def test_lleva_particion_y_dispositivo_en_el_nombre(self) -> None:
        """El nombre se lee sin abrir el fichero."""
        nombre = self._ruta(split="test").name
        assert nombre == (
            "s_ptq_int8-noexcl_5bcd71f0_onnxruntime_int8_test_640_workstation_cpu.json"
        )

    def test_no_coincide_con_el_nombre_antiguo(self) -> None:
        """Los 55 volcados anteriores no se renombran y no pueden ser pisados."""
        antiguo = "s_ptq_int8-noexcl_5bcd71f0_onnxruntime_int8_val_640.json"
        assert self._ruta(split="val").name != antiguo

    def test_un_dispositivo_con_barra_falla(self) -> None:
        """Un device_id con una barra no puede escribir en otro directorio."""
        from tfm_edge.common.exceptions import InvalidPathComponentError

        with pytest.raises(InvalidPathComponentError):
            self._ruta(device_id="workstation/cpu")


class TestHuellaDelArtefacto:
    """El volcado guarda el sha256 de lo que se midió."""

    _script = staticmethod(TestDondeVanLasDeteccionesCrudas._script)

    def test_un_onnx_es_un_fichero(self, tmp_path) -> None:
        """Bytes y sha256 del único fichero."""
        import hashlib

        onnx = tmp_path / "m.onnx"
        onnx.write_bytes(b"onnx")
        huella = self._script().artifact_files_fingerprint(onnx)
        assert huella == [
            {"file": "m.onnx", "bytes": 4, "sha256": hashlib.sha256(b"onnx").hexdigest()}
        ]

    def test_un_ir_son_dos_ficheros_por_xml_o_por_directorio(self, tmp_path) -> None:
        """El .bin determina el modelo tanto como el .xml."""
        ir = tmp_path / "ir"
        ir.mkdir()
        (ir / "m.xml").write_bytes(b"xml")
        (ir / "m.bin").write_bytes(b"pesos")
        script = self._script()
        por_xml = script.artifact_files_fingerprint(ir / "m.xml")
        por_dir = script.artifact_files_fingerprint(ir)
        assert [e["file"] for e in por_xml] == ["m.xml", "m.bin"]
        assert sorted(por_xml, key=lambda e: e["file"]) == por_dir

    def test_un_ir_sin_bin_falla(self, tmp_path) -> None:
        """Una huella a medias diría que se midió algo que no se pudo cargar."""
        (tmp_path / "m.xml").write_bytes(b"xml")
        with pytest.raises(FileNotFoundError):
            self._script().artifact_files_fingerprint(tmp_path / "m.xml")
