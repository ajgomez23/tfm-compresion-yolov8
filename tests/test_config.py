"""Tests del módulo de configuración."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tfm_edge.common import config as config_mod
from tfm_edge.common.config import (
    EXPERIMENT_IDENTITY_SECTIONS,
    BenchmarkConfig,
    DistillationConfig,
    ProjectConfig,
    PruningConfig,
    QuantizationConfig,
    SplitRatios,
    apply_overrides,
    compute_config_hash,
    compute_experiment_hash,
    deep_merge,
    dump_effective_config,
    experiment_identity_payload,
    load_class_names,
    load_config,
    load_yaml_with_inheritance,
    parse_override,
    validate_config,
)
from tfm_edge.common.exceptions import (
    ConfigFileNotFoundError,
    ConfigInheritanceError,
    ConfigOverrideError,
    ConfigSchemaError,
)
from tfm_edge.common.paths import get_project_paths

# ---------------------------------------------------------------------------
# Fusión en profundidad
# ---------------------------------------------------------------------------


class TestDeepMerge:
    """Comportamiento de la fusión en profundidad."""

    def test_fusiona_diccionarios_anidados(self) -> None:
        """Las claves anidadas se fusionan sin perder las hermanas."""
        base = {"training": {"epochs": 100, "batch_size": 16}, "seed": 42}
        override = {"training": {"epochs": 200}}
        resultado = deep_merge(base, override)
        assert resultado == {"training": {"epochs": 200, "batch_size": 16}, "seed": 42}

    def test_las_listas_se_reemplazan_no_se_concatenan(self) -> None:
        """Una lista heredada se sustituye entera."""
        base = {"export": {"formats": ["onnx", "tensorrt", "openvino"]}}
        override = {"export": {"formats": ["onnx"]}}
        assert deep_merge(base, override) == {"export": {"formats": ["onnx"]}}

    def test_no_modifica_los_argumentos(self) -> None:
        base = {"a": {"b": 1}}
        override = {"a": {"c": 2}}
        deep_merge(base, override)
        assert base == {"a": {"b": 1}}
        assert override == {"a": {"c": 2}}


# ---------------------------------------------------------------------------
# Herencia con _base_
# ---------------------------------------------------------------------------


class TestHerencia:
    """Carga de YAML con la clave ``_base_``."""

    def test_hereda_de_una_base(self, escribir_yaml) -> None:
        """Los valores no redefinidos se heredan de la base."""
        escribir_yaml("base.yaml", "seed: 42\ntraining:\n  epochs: 100\n  batch_size: 16\n")
        hijo = escribir_yaml("hijo.yaml", "_base_: base.yaml\ntraining:\n  epochs: 5\n")
        datos = load_yaml_with_inheritance(hijo)
        assert datos["seed"] == 42
        assert datos["training"]["epochs"] == 5
        assert datos["training"]["batch_size"] == 16

    def test_la_clave_base_no_sobrevive_a_la_carga(self, escribir_yaml) -> None:
        escribir_yaml("base.yaml", "seed: 1\n")
        hijo = escribir_yaml("hijo.yaml", "_base_: base.yaml\n")
        assert "_base_" not in load_yaml_with_inheritance(hijo)

    def test_hereda_de_varias_bases_en_orden(self, escribir_yaml) -> None:
        """Con una lista de bases, la última gana sobre la anterior."""
        escribir_yaml("a.yaml", "valor: 'a'\nsolo_a: 1\n")
        escribir_yaml("b.yaml", "valor: 'b'\nsolo_b: 2\n")
        hijo = escribir_yaml("hijo.yaml", "_base_:\n  - a.yaml\n  - b.yaml\n")
        datos = load_yaml_with_inheritance(hijo)
        assert datos == {"valor": "b", "solo_a": 1, "solo_b": 2}

    def test_herencia_encadenada_en_varios_niveles(self, escribir_yaml) -> None:
        escribir_yaml("abuelo.yaml", "a: 1\nb: 1\nc: 1\n")
        escribir_yaml("padre.yaml", "_base_: abuelo.yaml\nb: 2\nc: 2\n")
        hijo = escribir_yaml("hijo.yaml", "_base_: padre.yaml\nc: 3\n")
        assert load_yaml_with_inheritance(hijo) == {"a": 1, "b": 2, "c": 3}

    def test_las_rutas_base_son_relativas_al_fichero(self, escribir_yaml) -> None:
        """``_base_`` se resuelve respecto al fichero que la declara."""
        escribir_yaml("comun/base.yaml", "seed: 7\n")
        hijo = escribir_yaml("variantes/hijo.yaml", "_base_: ../comun/base.yaml\n")
        assert load_yaml_with_inheritance(hijo)["seed"] == 7

    def test_detecta_herencia_ciclica(self, escribir_yaml) -> None:
        """Un ciclo de herencia se detecta en lugar de agotar la pila."""
        escribir_yaml("a.yaml", "_base_: b.yaml\n")
        escribir_yaml("b.yaml", "_base_: a.yaml\n")
        with pytest.raises(ConfigInheritanceError, match="cíclica"):
            load_yaml_with_inheritance(escribir_yaml("a.yaml", "_base_: b.yaml\n"))

    def test_base_inexistente_da_error_util(self, escribir_yaml) -> None:
        """El error de base ausente indica desde dónde se la referenciaba."""
        hijo = escribir_yaml("hijo.yaml", "_base_: no_existe.yaml\n")
        with pytest.raises(ConfigFileNotFoundError, match="referenciado desde"):
            load_yaml_with_inheritance(hijo)

    def test_fichero_vacio_es_configuracion_vacia(self, escribir_yaml) -> None:
        assert load_yaml_with_inheritance(escribir_yaml("vacio.yaml", "")) == {}

    def test_raiz_que_no_es_diccionario_es_error(self, escribir_yaml) -> None:
        """Una lista en la raíz del YAML no es una configuración válida."""
        with pytest.raises(ConfigSchemaError, match="diccionario"):
            load_yaml_with_inheritance(escribir_yaml("lista.yaml", "- a\n- b\n"))


# ---------------------------------------------------------------------------
# Validación de esquema
# ---------------------------------------------------------------------------


class TestValidacion:
    """Validación con Pydantic y rechazo de claves desconocidas."""

    def test_configuracion_por_defecto_es_valida(self) -> None:
        """Un diccionario vacío produce la configuración por defecto."""
        configuracion = validate_config({})
        assert configuracion.seed == 42
        assert configuracion.model.family == "yolov8"
        assert configuracion.technique == "baseline"

    def test_rechaza_clave_desconocida_en_la_raiz(self) -> None:
        """Una clave inexistente falla en lugar de ignorarse."""
        with pytest.raises(ConfigSchemaError) as excinfo:
            validate_config({"clave_inventada": 1})
        assert "clave_inventada" in str(excinfo.value)

    def test_rechaza_clave_desconocida_anidada(self) -> None:
        """El rechazo de claves desconocidas también aplica dentro de secciones."""
        with pytest.raises(ConfigSchemaError) as excinfo:
            validate_config({"training": {"epocs": 50}})
        mensaje = str(excinfo.value)
        assert "training.epocs" in mensaje

    def test_el_mensaje_de_error_indica_la_ruta_de_la_clave(self) -> None:
        """El error señala la ruta anidada exacta del problema."""
        with pytest.raises(ConfigSchemaError) as excinfo:
            validate_config({"model": {"input_resolution": 100}})
        assert "model.input_resolution" in str(excinfo.value)

    def test_resolucion_debe_ser_multiplo_de_32(self) -> None:
        """La resolución de entrada debe ser múltiplo de 32."""
        with pytest.raises(ConfigSchemaError, match="múltiplo de 32"):
            validate_config({"model": {"input_resolution": 100}})
        assert validate_config({"model": {"input_resolution": 640}}).model.input_resolution == 640

    def test_las_proporciones_de_particion_deben_sumar_uno(self) -> None:
        with pytest.raises(ValueError, match="sumar 1.0"):
            SplitRatios(train=0.7, val=0.2, test=0.2)
        assert SplitRatios(train=0.7, val=0.15, test=0.15).train == pytest.approx(0.7)

    def test_ratio_de_poda_fuera_de_rango(self) -> None:
        """Un ratio de poda de 1.0 dejaría la red sin canales."""
        with pytest.raises(ConfigSchemaError):
            validate_config({"pruning": {"ratio": 1.0}})

    def test_reparto_por_bloques_exige_ratios(self) -> None:
        """``per_block`` sin ``block_ratios`` es incoherente."""
        with pytest.raises(ValueError, match="block_ratios"):
            PruningConfig(distribution="per_block")

    def test_ratios_por_bloque_sin_reparto_por_bloques(self) -> None:
        """Definir ratios que se ignorarían en silencio es un error."""
        with pytest.raises(ValueError, match="se ignorarían"):
            PruningConfig(distribution="global", block_ratios={"backbone": 0.5})

    def test_destilacion_de_caracteristicas_exige_pares(self) -> None:
        """La destilación por características necesita pares de capas."""
        with pytest.raises(ValueError, match="feature_pairs"):
            DistillationConfig(kd_type="feature")

    @pytest.mark.parametrize("bandera", ["bias_correction", "cross_layer_equalization"])
    def test_las_banderas_no_implementadas_se_rechazan(self, bandera: str) -> None:
        with pytest.raises(ValueError, match=bandera):
            QuantizationConfig(**{bandera: True})

    def test_el_rechazo_llega_tambien_desde_el_yaml(self) -> None:
        with pytest.raises(ConfigSchemaError, match="bias_correction"):
            validate_config({"quantization": {"bias_correction": True}})

    def test_las_banderas_siguen_serializandose_en_false(self) -> None:
        """La guarda no puede mover el hash de las campañas ya inscritas."""
        volcado = QuantizationConfig().model_dump()
        assert volcado["bias_correction"] is False
        assert volcado["cross_layer_equalization"] is False

    def test_las_secciones_de_compresion_son_opcionales(self) -> None:
        """Un baseline no declara poda, y su sección queda a None."""
        configuracion = validate_config({})
        assert configuracion.pruning is None
        assert configuracion.quantization is None
        assert configuracion.distillation is None

    def test_valores_por_defecto_del_protocolo_de_medida(self) -> None:
        """Los valores por defecto son los que fija la sección 12 de la especificación."""
        benchmark = BenchmarkConfig()
        assert benchmark.warmup_iters == 50
        assert benchmark.measure_iters == 300
        assert benchmark.num_repeats == 3
        assert benchmark.batch_size == 1


# ---------------------------------------------------------------------------
# Overrides de línea de comandos
# ---------------------------------------------------------------------------


class TestOverrides:
    """Interpretación y aplicación de los overrides ``clave=valor``."""

    @pytest.mark.parametrize(
        ("texto", "claves", "valor"),
        [
            ("training.epochs=150", ["training", "epochs"], 150),
            ("pruning.ratio=0.6", ["pruning", "ratio"], 0.6),
            ("training.amp=false", ["training", "amp"], False),
            ("seed=0", ["seed"], 0),
            ("benchmark.num_threads=null", ["benchmark", "num_threads"], None),
            ("export.formats=[onnx, tflite]", ["export", "formats"], ["onnx", "tflite"]),
            ("notes=texto libre", ["notes"], "texto libre"),
        ],
    )
    def test_interpreta_el_tipo_del_valor(self, texto: str, claves: list[str], valor) -> None:
        assert parse_override(texto) == (claves, valor)

    def test_permite_forzar_una_cadena_con_comillas(self) -> None:
        """Entrecomillar preserva el valor como texto."""
        assert parse_override("technique_variant='150'") == (["technique_variant"], "150")

    def test_el_valor_puede_contener_signos_igual(self) -> None:
        claves, valor = parse_override("notes=a=b=c")
        assert claves == ["notes"]
        assert valor == "a=b=c"

    @pytest.mark.parametrize("texto", ["sin_igual", "=sin_clave", "a..b=1"])
    def test_rechaza_overrides_mal_formados(self, texto: str) -> None:
        """Un override mal formado falla con un mensaje explicativo."""
        with pytest.raises(ConfigOverrideError):
            parse_override(texto)

    def test_aplica_override_anidado(self) -> None:
        """El override llega a una clave anidada sin borrar sus hermanas."""
        datos = {"training": {"epochs": 100, "batch_size": 16}}
        resultado = apply_overrides(datos, ["training.epochs=5"])
        assert resultado["training"] == {"epochs": 5, "batch_size": 16}

    def test_crea_las_claves_intermedias_que_falten(self) -> None:
        """Un override puede crear una sección que no existía."""
        resultado = apply_overrides({}, ["pruning.ratio=0.25"])
        assert resultado == {"pruning": {"ratio": 0.25}}

    def test_falla_si_una_clave_intermedia_no_es_diccionario(self) -> None:
        """No se puede descender por una clave escalar."""
        with pytest.raises(ConfigOverrideError, match="diccionario"):
            apply_overrides({"seed": 42}, ["seed.sub=1"])

    def test_no_modifica_el_diccionario_original(self) -> None:
        """Aplicar overrides es una operación pura."""
        datos = {"training": {"epochs": 100}}
        apply_overrides(datos, ["training.epochs=1"])
        assert datos["training"]["epochs"] == 100

    def test_un_override_desconocido_falla_al_validar(self) -> None:
        """Un override con una clave inexistente lo caza el esquema."""
        with pytest.raises(ConfigSchemaError):
            validate_config(apply_overrides({}, ["training.no_existe=1"]))


# ---------------------------------------------------------------------------
# Carga completa
# ---------------------------------------------------------------------------


class TestCargaCompleta:
    """Carga completa: fichero, herencia, overrides y validación."""

    def test_carga_hereda_sobrescribe_y_valida(self, escribir_yaml) -> None:
        """El camino completo produce una configuración válida y coherente."""
        escribir_yaml("base.yaml", "seed: 42\ntraining:\n  epochs: 100\n  batch_size: 16\n")
        hijo = escribir_yaml(
            "variante.yaml",
            "_base_: base.yaml\ntechnique: pruning\ntechnique_variant: l1global_r50\n"
            "pruning:\n  criterion: l1\n  ratio: 0.5\n",
        )
        configuracion = load_config(hijo, overrides=["training.epochs=3"])
        assert configuracion.seed == 42
        assert configuracion.training.epochs == 3
        assert configuracion.training.batch_size == 16
        assert configuracion.pruning is not None
        assert configuracion.pruning.ratio == pytest.approx(0.5)

    def test_fichero_inexistente(self, tmp_path: Path) -> None:
        """Cargar un fichero que no existe falla con un mensaje claro."""
        with pytest.raises(ConfigFileNotFoundError):
            load_config(tmp_path / "no_existe.yaml")

    def test_el_default_yaml_del_repositorio_es_valido(self) -> None:
        """configs/default.yaml debe validar contra el esquema."""
        from tfm_edge.common.paths import get_project_paths

        ruta = get_project_paths().configs / "default.yaml"
        if not ruta.is_file():
            pytest.skip("configs/default.yaml no está presente en este árbol")
        configuracion = load_config(ruta)
        assert configuracion.model.family == "yolov8"
        assert configuracion.benchmark.measure_iters == 300


# ---------------------------------------------------------------------------
# Hash y serialización
# ---------------------------------------------------------------------------


class TestHash:
    """Estabilidad y alcance del hash de configuración."""

    def test_es_estable_entre_llamadas(self) -> None:
        """El mismo contenido produce siempre el mismo hash."""
        payload = {"b": 2, "a": 1}
        assert compute_config_hash(payload) == compute_config_hash(payload)

    def test_no_depende_del_orden_de_las_claves(self) -> None:
        """Dos diccionarios equivalentes producen el mismo hash."""
        assert compute_config_hash({"a": 1, "b": 2}) == compute_config_hash({"b": 2, "a": 1})

    def test_cambia_al_cambiar_un_valor(self) -> None:
        """Un cambio en la configuración cambia el hash."""
        assert compute_config_hash({"a": 1}) != compute_config_hash({"a": 2})

    def test_longitud_por_defecto(self) -> None:
        """El hash corto tiene ocho caracteres hexadecimales."""
        valor = compute_config_hash({"a": 1})
        assert len(valor) == 8
        assert all(caracter in "0123456789abcdef" for caracter in valor)

    def test_longitud_fuera_de_rango(self) -> None:
        """Una longitud imposible es un error explícito."""
        with pytest.raises(ValueError):
            compute_config_hash({"a": 1}, length=0)

    def test_la_identidad_excluye_benchmark_export_y_evaluation(self) -> None:
        """Medir el mismo modelo en otro sitio no lo convierte en otro modelo."""
        base = validate_config({})
        otro_dispositivo = validate_config(
            {"benchmark": {"device_id": "edge_cpu_arm", "measure_iters": 42}}
        )
        assert compute_experiment_hash(base) == compute_experiment_hash(otro_dispositivo)

    def test_la_identidad_incluye_el_modelo_y_el_entrenamiento(self) -> None:
        """Cambiar el modelo o el entrenamiento sí cambia la identidad."""
        base = validate_config({})
        assert compute_experiment_hash(base) != compute_experiment_hash(
            validate_config({"model": {"size": "m"}})
        )
        assert compute_experiment_hash(base) != compute_experiment_hash(
            validate_config({"training": {"epochs": 7}})
        )

    def test_la_identidad_incluye_la_compresion(self) -> None:
        """Dos podas con ratios distintos son experimentos distintos."""
        poda_25 = validate_config({"pruning": {"ratio": 0.25}})
        poda_50 = validate_config({"pruning": {"ratio": 0.50}})
        assert compute_experiment_hash(poda_25) != compute_experiment_hash(poda_50)

    def test_el_payload_de_identidad_solo_trae_secciones_de_identidad(self) -> None:
        """El payload no contiene ninguna sección ajena a la identidad."""
        payload = experiment_identity_payload(validate_config({}))
        assert set(payload).issubset(set(EXPERIMENT_IDENTITY_SECTIONS))
        for excluida in ("benchmark", "export", "evaluation"):
            assert excluida not in payload


class TestSerializacion:
    """Volcado de la configuración efectiva junto a los resultados."""

    def test_vuelca_y_recarga_sin_perder_informacion(self, tmp_path: Path) -> None:
        """La configuración efectiva volcada se puede recargar idéntica."""
        original = validate_config({"training": {"epochs": 7}, "pruning": {"ratio": 0.3}})
        destino = dump_effective_config(original, tmp_path / "sub" / "efectiva.yaml")
        assert destino.is_file()
        recargada = load_config(destino)
        assert recargada.model_dump() == original.model_dump()

    def test_crea_los_directorios_padre(self, tmp_path: Path) -> None:
        """El volcado crea el árbol de directorios que falte."""
        destino = dump_effective_config(ProjectConfig(), tmp_path / "a" / "b" / "c.yaml")
        assert destino.is_file()

    def test_el_volcado_conserva_el_hash_de_identidad(self, tmp_path: Path) -> None:
        original = validate_config({"model": {"size": "m"}})
        recargada = load_config(dump_effective_config(original, tmp_path / "e.yaml"))
        assert compute_experiment_hash(recargada) == compute_experiment_hash(original)


# ---------------------------------------------------------------------------
# Mapeo canónico de clases
# ---------------------------------------------------------------------------


class TestNombresDeClase:
    """Carga del JSON canónico de nombres de clase."""

    def test_carga_la_lista(self, tmp_path: Path) -> None:
        """Un JSON bien formado devuelve la lista en orden."""
        fichero = tmp_path / "clases.json"
        fichero.write_text('{"classes": ["persona", "casco"]}', encoding="utf-8")
        assert load_class_names(fichero) == ["persona", "casco"]

    def test_fichero_ausente_es_error_no_valor_por_defecto(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigFileNotFoundError):
            load_class_names(tmp_path / "no_existe.json")

    @pytest.mark.parametrize(
        "contenido",
        ['{"otra_clave": []}', '{"classes": []}', '{"classes": [1, 2]}', "[]"],
    )
    def test_rechaza_formatos_invalidos(self, tmp_path: Path, contenido: str) -> None:
        """Un JSON con formato inesperado falla con un mensaje claro."""
        fichero = tmp_path / "clases.json"
        fichero.write_text(contenido, encoding="utf-8")
        with pytest.raises(ConfigSchemaError):
            load_class_names(fichero)

    def test_el_mapeo_canonico_de_sh17_es_coherente(self) -> None:
        """El JSON canónico de SH17 concuerda con la configuración del dataset."""
        from tfm_edge.common.paths import get_project_paths

        rutas = get_project_paths()
        fichero = rutas.configs / "dataset" / "sh17_classes.json"
        if not fichero.is_file():
            pytest.skip("configs/dataset/sh17_classes.json no está presente")

        clases = load_class_names(fichero)
        assert len(clases) == 17
        # Las cuatro primeras del orden autoritativo de sh17.yaml. Si alguien
        # sustituyera el mapeo por el del README, esto fallaría de inmediato.
        assert clases[:4] == ["person", "ear", "ear-mufs", "face"]

        configuracion = load_config(rutas.configs / "dataset" / "sh17.yaml")
        assert configuracion.dataset.num_classes == len(clases)


class TestResolucionUnicaFuenteDeVerdad:
    """La resolución de entrada es configurable y tiene una sola fuente."""

    def test_solo_model_declara_la_resolucion(self) -> None:
        """Ninguna otra sección declara su propia resolución de entrada."""
        configuracion = validate_config({})
        sospechosos = ("resolution", "imgsz", "img_size", "input_size")
        for seccion in ("export", "evaluation", "benchmark", "training", "dataset"):
            campos = type(getattr(configuracion, seccion)).model_fields
            encontrados = [c for c in campos if any(s in c for s in sospechosos)]
            assert not encontrados, (
                f"La sección '{seccion}' declara {encontrados}, que compite con "
                "model.input_resolution como fuente de la resolución."
            )
        assert "input_resolution" in type(configuracion.model).model_fields

    def test_la_resolucion_es_configurable_y_cambia_la_identidad(self) -> None:
        base = validate_config({})
        reducida = validate_config({"model": {"input_resolution": 416}})
        assert reducida.model.input_resolution == 416
        assert compute_experiment_hash(base) != compute_experiment_hash(reducida)


class TestLasOpcionesQueNadieLee:
    """Opciones del esquema que ningún código lee."""

    def test_todos_los_campos_de_la_tabla_existen(self) -> None:
        """Un nombre mal escrito dejaría la guarda mirando un campo inexistente."""
        import tfm_edge.common.config as modulo

        for nombre_modelo, campos in config_mod.OPCIONES_SIN_LECTOR.items():
            modelo = getattr(modulo, nombre_modelo)
            desconocidos = sorted(set(campos) - set(modelo.model_fields))
            assert not desconocidos, f"{nombre_modelo} no declara {desconocidos}"

    def test_ninguna_ha_ganado_un_lector_sin_que_la_tabla_se_entere(self) -> None:
        """La mitad simétrica, que es la que caduca sola."""
        raiz = get_project_paths().root
        fuentes = [
            ruta
            for patron in ("src/**/*.py", "scripts/**/*.py")
            for ruta in raiz.glob(patron)
            if "__pycache__" not in str(ruta) and ruta.name != "config.py"
        ]
        textos = [ruta.read_text(encoding="utf-8") for ruta in fuentes]

        con_lector = []
        for campos in config_mod.OPCIONES_SIN_LECTOR.values():
            for campo in campos:
                patron = re.compile(rf"(\.{re.escape(campo)}\b|[\"\']{re.escape(campo)}[\"\'])")
                if any(patron.search(contenido) for contenido in textos):
                    con_lector.append(campo)

        assert not sorted(set(con_lector)), (
            "Estas opciones ya las lee alguien y siguen en OPCIONES_SIN_LECTOR: "
            f"{sorted(set(con_lector))}. Si se han implementado, hay que sacarlas de "
            "la tabla; si no, hay que mirar qué las lee."
        )

    def test_cada_motivo_dice_que_hace_el_codigo(self) -> None:
        cortos = {
            f"{modelo}.{campo}": motivo
            for modelo, campos in config_mod.OPCIONES_SIN_LECTOR.items()
            for campo, motivo in campos.items()
            if len(motivo) < 30
        }

        assert not cortos, f"motivos que no explican nada: {cortos}"

    @pytest.mark.parametrize(
        "seccion, campo, valor",
        [
            ("quantization", "weight_granularity", "per_tensor"),
            ("quantization", "bias_correction", True),
            ("export", "tolerance_fp32", 0.5),
            ("benchmark", "throughput_window_s", 60.0),
            ("dataset", "stratified", False),
        ],
    )
    def test_pedir_un_valor_distinto_falla(self, seccion: str, campo: str, valor: object) -> None:
        with pytest.raises(ConfigSchemaError, match=campo):
            validate_config({seccion: {campo: valor}})

    def test_los_yaml_del_proyecto_siguen_cargando(self) -> None:
        raiz = get_project_paths().root
        yamls = [
            ruta
            for ruta in sorted((raiz / "configs").rglob("*.yaml"))
            if ruta.parent.name != "benchmark"
        ]
        assert len(yamls) >= 15, f"solo {len(yamls)} YAML: el barrido no está mirando nada"

        for ruta in yamls:
            load_config(ruta)
