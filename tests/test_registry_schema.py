"""Tests del registro único de resultados y de la convención de identificadores."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tfm_edge.common.environment import (
    ENVIRONMENT_COLUMNS,
    ENVIRONMENT_SCHEMA_EXTENSIONS,
    compute_environment_id,
)
from tfm_edge.common.exceptions import (
    InvalidExperimentIdError,
    InvalidPathComponentError,
    RegistryDuplicateError,
    RegistryIntegrityError,
    RegistrySchemaError,
)
from tfm_edge.common.paths import (
    build_experiment_id,
    get_project_paths,
    parse_experiment_id,
    slugify,
    validate_path_component,
)
from tfm_edge.common.registry import (
    DEPLOYMENT_COLUMNS,
    DEPLOYMENT_SCHEMA_EXTENSIONS,
    EXPERIMENTS_COLUMNS,
    VALID_STATUS,
    ExperimentRegistry,
    build_ap_class_columns,
    build_quality_columns,
    is_absolute_path_text,
    normalize_value,
    utc_timestamp,
)


def fila_experimento(experiment_id: str = "s_baseline_default_a1b2c3d4", **cambios) -> dict:
    """Construye una fila completa de experiments.csv para los tests."""
    fila = {columna: None for columna in EXPERIMENTS_COLUMNS}
    fila.update(
        {
            "experiment_id": experiment_id,
            "timestamp": utc_timestamp(),
            "model_family": "yolov8",
            "model_size": "s",
            "technique": "baseline",
            "technique_variant": "default",
            "config_hash": "a1b2c3d4",
            "status": "ok",
        }
    )
    fila.update(cambios)
    return fila


def fila_despliegue(experiment_id: str = "s_baseline_default_a1b2c3d4", **cambios) -> dict:
    """Construye una fila completa de deployment_metrics.csv para los tests."""
    fila = {columna: None for columna in DEPLOYMENT_COLUMNS}
    fila.update(
        {
            "experiment_id": experiment_id,
            "runtime": "pytorch",
            "precision": "fp32",
            "device_id": "workstation_cpu",
            "batch_size": 1,
            "timestamp": utc_timestamp(),
            "status": "ok",
        }
    )
    fila.update(cambios)
    return fila


# ---------------------------------------------------------------------------
# Convención de experiment_id (sección 16.2)
# ---------------------------------------------------------------------------


class TestExperimentId:
    """Construcción, validación y troceo del identificador de experimento."""

    def test_formato_basico(self) -> None:
        """El identificador se compone en el orden fijado por la especificación."""
        assert build_experiment_id("s", "baseline", "default", "a1b2c3d4") == (
            "s_baseline_default_a1b2c3d4"
        )

    def test_ejemplos_de_la_especificacion(self) -> None:
        """Los tres ejemplos literales de la sección 16.2 son válidos."""
        assert (
            build_experiment_id("s", "pruning", "l1global_r50", "9f8e7d6c")
            == "s_pruning_l1global_r50_9f8e7d6c"
        )
        assert (
            build_experiment_id("s", "combined", "prune50-ptqint8", "11aa22bb")
            == "s_combined_prune50-ptqint8_11aa22bb"
        )

    def test_recorta_el_hash_a_ocho_caracteres(self) -> None:
        """Se admite un hash largo y se recorta a la longitud canónica."""
        identificador = build_experiment_id("n", "ptq", "int8", "0123456789abcdef")
        assert identificador.endswith("_01234567")

    @pytest.mark.parametrize("tamano", ["x", "S", "xl", ""])
    def test_rechaza_tamanos_invalidos(self, tamano: str) -> None:
        """Solo se admiten los cuatro tamaños contemplados."""
        with pytest.raises(InvalidExperimentIdError):
            build_experiment_id(tamano, "baseline", "default", "a1b2c3d4")

    @pytest.mark.parametrize("tecnica", ["poda", "PTQ", "quantization", ""])
    def test_rechaza_tecnicas_invalidas(self, tecnica: str) -> None:
        """La técnica debe pertenecer al conjunto cerrado de la sección 16.1."""
        with pytest.raises(InvalidExperimentIdError):
            build_experiment_id("s", tecnica, "default", "a1b2c3d4")

    @pytest.mark.parametrize("hash_malo", ["xyz12345", "abc", "", "12 34567"])
    def test_rechaza_hashes_invalidos(self, hash_malo: str) -> None:
        """El hash debe ser hexadecimal y de ocho caracteres."""
        with pytest.raises(InvalidExperimentIdError):
            build_experiment_id("s", "baseline", "default", hash_malo)

    def test_normaliza_el_hash_a_minusculas(self) -> None:
        """Un hash en mayúsculas se normaliza en lugar de rechazarse."""
        assert (
            build_experiment_id("s", "baseline", "default", "A1B2C3D4")
            == "s_baseline_default_a1b2c3d4"
        )

    @pytest.mark.parametrize("variante", ["con espacio", "con/barra", "-empieza-por-guion", ""])
    def test_rechaza_variantes_invalidas(self, variante: str) -> None:
        """La variante no puede romper la validez del nombre de fichero."""
        with pytest.raises(InvalidExperimentIdError):
            build_experiment_id("s", "baseline", variante, "a1b2c3d4")

    def test_troceo_posicional_admite_guiones_bajos_en_la_variante(self) -> None:
        """El troceo es posicional, así que la variante puede llevar '_'."""
        partes = parse_experiment_id("s_pruning_l1global_r50_9f8e7d6c")
        assert partes == {
            "model_size": "s",
            "technique": "pruning",
            "variant": "l1global_r50",
            "config_hash": "9f8e7d6c",
        }

    def test_ida_y_vuelta(self) -> None:
        """Construir y trocear son operaciones inversas."""
        original = build_experiment_id("m", "distillation", "kd_combined_t4", "deadbeef")
        partes = parse_experiment_id(original)
        assert build_experiment_id(**partes) == original

    @pytest.mark.parametrize(
        "identificador", ["s_baseline_a1b2c3d4", "s_desconocida_v_a1b2c3d4", "s_baseline_v_XYZ"]
    )
    def test_rechaza_identificadores_mal_formados(self, identificador: str) -> None:
        """Un identificador que no sigue la convención se detecta al trocearlo."""
        with pytest.raises(InvalidExperimentIdError):
            parse_experiment_id(identificador)


class TestValidezDeNombresDeFichero:
    """Portabilidad de los identificadores entre Linux y Windows."""

    @pytest.mark.parametrize("nombre", ["CON", "nul", "COM1", "LPT9", "aux.txt"])
    def test_rechaza_nombres_reservados_de_windows(self, nombre: str) -> None:
        """Los nombres reservados de Windows no son utilizables como directorio."""
        with pytest.raises(InvalidPathComponentError, match="reservado"):
            validate_path_component(nombre)

    @pytest.mark.parametrize("nombre", ["a<b", "a>b", 'a"b', "a|b", "a?b", "a*b", "a:b", "a/b"])
    def test_rechaza_caracteres_prohibidos(self, nombre: str) -> None:
        """Los caracteres prohibidos en Windows se rechazan también en Linux."""
        with pytest.raises(InvalidPathComponentError, match="prohibidos"):
            validate_path_component(nombre)

    @pytest.mark.parametrize("nombre", ["termina.", "termina "])
    def test_rechaza_punto_o_espacio_final(self, nombre: str) -> None:
        """Windows recorta el punto y el espacio finales, cambiando el nombre."""
        with pytest.raises(InvalidPathComponentError, match="termina"):
            validate_path_component(nombre)

    def test_rechaza_nombre_vacio_y_demasiado_largo(self) -> None:
        """Ni vacío ni desmesurado."""
        with pytest.raises(InvalidPathComponentError):
            validate_path_component("")
        with pytest.raises(InvalidPathComponentError):
            validate_path_component("a" * 500)

    def test_acepta_un_identificador_real(self) -> None:
        """Un identificador construido por la convención siempre es válido."""
        assert validate_path_component("s_pruning_l1global_r50_9f8e7d6c")


class TestSlugify:
    """Normalización de nombres para las cabeceras del registro."""

    @pytest.mark.parametrize(
        ("entrada", "esperado"),
        [
            ("Persona", "persona"),
            ("casco de seguridad", "casco_de_seguridad"),
            ("Gafas-protectoras", "gafas_protectoras"),
            ("máscara", "mascara"),
            ("  espacios  ", "espacios"),
            ("Múltiples   espacios", "multiples_espacios"),
        ],
    )
    def test_normaliza(self, entrada: str, esperado: str) -> None:
        """Los acentos, mayúsculas y separadores se normalizan de forma estable."""
        assert slugify(entrada) == esperado


# ---------------------------------------------------------------------------
# Esquema de columnas (sección 16)
# ---------------------------------------------------------------------------


class TestEsquemaDeColumnas:
    """El esquema declarado coincide con el de la especificación."""

    def test_experiments_tiene_las_columnas_de_la_seccion_16_1(self) -> None:
        """Comprobación literal contra la lista de la especificación."""
        esperadas = (
            "experiment_id parent_experiment_id timestamp model_family model_size technique "
            "technique_variant config_hash config_path compression_ratio effective_params "
            "effective_params_ratio precision training_epochs training_time_s training_device "
            "weights_path status notes"
        ).split()
        assert list(EXPERIMENTS_COLUMNS) == esperadas

    def test_deployment_incluye_todas_las_medidas_exigidas(self) -> None:
        """Los percentiles, los tres tiempos y las tres memorias están presentes."""
        for columna in (
            "latency_p50_ms",
            "latency_p90_ms",
            "latency_p95_ms",
            "latency_p99_ms",
            "preprocess_mean_ms",
            "inference_mean_ms",
            "postprocess_mean_ms",
            "peak_gpu_memory_mb",
            "peak_host_memory_mb",
            "model_load_memory_mb",
            "artifact_size_bytes",
            "artifact_size_gzip_bytes",
            "raw_timings_path",
        ):
            assert columna in DEPLOYMENT_COLUMNS

    def test_environments_tiene_las_columnas_de_la_seccion_16_5(self) -> None:
        """Comprobación literal contra la lista de la especificación."""
        esperadas = (
            "environment_id device_id os os_version architecture cpu_model cpu_cores_physical "
            "cpu_cores_logical ram_total_gb gpu_model gpu_vram_gb driver_version cuda_version "
            "cudnn_version python_version torch_version ultralytics_version onnxruntime_version "
            "tensorrt_version openvino_version tflite_version git_commit git_dirty timestamp"
        ).split()
        sin_extensiones = [c for c in ENVIRONMENT_COLUMNS if c not in ENVIRONMENT_SCHEMA_EXTENSIONS]
        assert sin_extensiones == esperadas
        assert set(ENVIRONMENT_SCHEMA_EXTENSIONS).issubset(set(ENVIRONMENT_COLUMNS))

    def test_deployment_solo_extiende_16_4_de_forma_declarada(self) -> None:
        """deployment_metrics solo puede desviarse de 16.4 por extensión declarada."""
        esperadas = (
            "experiment_id runtime precision device_id environment_id input_resolution "
            "batch_size num_threads execution_provider warmup_iters measure_iters num_repeats "
            "latency_mean_ms latency_std_ms latency_min_ms latency_max_ms latency_p50_ms "
            "latency_p90_ms latency_p95_ms latency_p99_ms preprocess_mean_ms inference_mean_ms "
            "postprocess_mean_ms throughput_img_s peak_gpu_memory_mb peak_host_memory_mb "
            "model_load_memory_mb artifact_size_bytes artifact_size_gzip_bytes artifact_path "
            "export_time_s engine_build_time_s temperature_start_c temperature_end_c "
            "raw_timings_path timestamp status notes"
        ).split()
        sin_extensiones = [c for c in DEPLOYMENT_COLUMNS if c not in DEPLOYMENT_SCHEMA_EXTENSIONS]
        assert sin_extensiones == esperadas

    def test_columnas_de_ap_por_clase(self, class_names: list[str]) -> None:
        """Las columnas por clase llevan índice y nombre normalizado."""
        columnas = build_ap_class_columns(class_names)
        assert len(columnas) == len(class_names)
        assert columnas[0] == "ap_class_00_persona"
        assert columnas[1] == "ap_class_01_casco_de_seguridad"
        assert columnas[6] == "ap_class_06_mascara"

    def test_las_columnas_por_clase_van_en_su_sitio(self, class_names: list[str]) -> None:
        """El bloque por clase va entre f1_best y error_classification (16.3)."""
        columnas = build_quality_columns(class_names)
        assert columnas.index("f1_best") < columnas.index("ap_class_00_persona")
        assert columnas.index("ap_class_06_mascara") < columnas.index("error_classification")
        assert columnas[-1] == "notes"

    def test_rechaza_lista_de_clases_vacia(self) -> None:
        """Sin nombres de clase no se puede construir el esquema."""
        with pytest.raises(RegistrySchemaError):
            build_ap_class_columns([])

    def test_rechaza_nombres_de_clase_ambiguos(self) -> None:
        """Dos clases que normalizan igual harían ambiguas las columnas."""
        with pytest.raises(RegistrySchemaError, match="ambiguas"):
            build_ap_class_columns(["Casco", "casco"])


# ---------------------------------------------------------------------------
# Normalización de valores (sección 16.6)
# ---------------------------------------------------------------------------


class TestNormalizacionDeValores:
    """Los valores ausentes se escriben como celda vacía, nunca como 0 o NaN."""

    def test_none_es_celda_vacia(self) -> None:
        """``None`` significa ausencia de medida."""
        assert normalize_value(None) == ""

    def test_nan_es_celda_vacia(self) -> None:
        """Un NaN de coma flotante también es ausencia."""
        assert normalize_value(float("nan")) == ""

    def test_el_cero_se_conserva(self) -> None:
        assert normalize_value(0) == "0"
        assert normalize_value(0.0) == "0.0"

    def test_booleanos_en_minuscula(self) -> None:
        """Los booleanos se escriben de forma consistente con git_dirty."""
        assert normalize_value(True) == "true"
        assert normalize_value(False) == "false"


# ---------------------------------------------------------------------------
# Escritura y lectura
# ---------------------------------------------------------------------------


class TestEscritura:
    """Escritura de filas, esquema y política de duplicados."""

    def test_escribe_y_relee_una_fila(self, registry_dir: Path) -> None:
        """Una fila escrita se recupera con los mismos valores."""
        registro = ExperimentRegistry(registry_dir)
        assert registro.append_experiment(fila_experimento()) is True
        filas = registro.read_rows("experiments")
        assert len(filas) == 1
        assert filas[0]["experiment_id"] == "s_baseline_default_a1b2c3d4"
        assert filas[0]["model_family"] == "yolov8"

    def test_la_cabecera_es_exactamente_el_esquema(self, registry_dir: Path) -> None:
        """El CSV escrito tiene las columnas del esquema, en orden."""
        registro = ExperimentRegistry(registry_dir)
        registro.append_experiment(fila_experimento())
        with (registry_dir / "experiments.csv").open(encoding="utf-8", newline="") as manejador:
            cabecera = next(csv.reader(manejador))
        assert tuple(cabecera) == EXPERIMENTS_COLUMNS

    def test_los_valores_ausentes_quedan_como_celda_vacia(self, registry_dir: Path) -> None:
        """Las columnas sin valor no se rellenan con ceros ni con 'NaN'."""
        registro = ExperimentRegistry(registry_dir)
        registro.append_experiment(fila_experimento())
        fila = registro.read_rows("experiments")[0]
        assert fila["training_time_s"] == ""
        assert fila["parent_experiment_id"] == ""

    def test_rechaza_columnas_sobrantes(self, registry_dir: Path) -> None:
        """Una columna inventada es un error de esquema."""
        registro = ExperimentRegistry(registry_dir)
        fila = fila_experimento()
        fila["columna_inventada"] = 1
        with pytest.raises(RegistrySchemaError, match="no reconocidas"):
            registro.append_experiment(fila)

    def test_rechaza_columnas_faltantes(self, registry_dir: Path) -> None:
        """Omitir una columna es distinto de dejarla vacía, y es un error."""
        registro = ExperimentRegistry(registry_dir)
        fila = fila_experimento()
        del fila["notes"]
        with pytest.raises(RegistrySchemaError, match="faltan columnas"):
            registro.append_experiment(fila)

    @pytest.mark.parametrize("estado", ["OK", "correcto", "failed", "error"])
    def test_rechaza_estados_no_contemplados(self, registry_dir: Path, estado: str) -> None:
        """Solo se admiten los tres estados de la especificación."""
        registro = ExperimentRegistry(registry_dir)
        with pytest.raises(RegistrySchemaError, match="status"):
            registro.append_experiment(fila_experimento(status=estado))

    @pytest.mark.parametrize("estado", VALID_STATUS)
    def test_admite_los_tres_estados(self, registry_dir: Path, estado: str) -> None:
        """ok, fallido y no_disponible son válidos."""
        registro = ExperimentRegistry(registry_dir)
        assert registro.append_experiment(fila_experimento(status=estado)) is True

    def test_rechaza_ausencias_codificadas_como_texto(self, registry_dir: Path) -> None:
        registro = ExperimentRegistry(registry_dir)
        with pytest.raises(RegistrySchemaError, match="ausencia"):
            registro.append_experiment(fila_experimento(training_time_s="NaN"))

    def test_las_tablas_de_metricas_necesitan_su_esquema(
        self, registry_dir: Path, class_names: list[str]
    ) -> None:
        """quality_metrics no se puede escribir sin los nombres de clase."""
        sin_clases = ExperimentRegistry(registry_dir)
        with pytest.raises(RegistrySchemaError, match="nombres de las clases"):
            sin_clases.columns_for("quality_metrics")
        con_clases = ExperimentRegistry(registry_dir, class_names=class_names)
        assert len(con_clases.columns_for("quality_metrics")) == (
            len(build_quality_columns(class_names))
        )

    def test_tabla_desconocida(self, registry_dir: Path) -> None:
        """Pedir una tabla inexistente falla explícitamente."""
        with pytest.raises(RegistrySchemaError, match="desconocida"):
            ExperimentRegistry(registry_dir).path_for("inventada")


class TestDuplicados:
    """Las cuatro políticas ante una clave primaria repetida."""

    def test_skip_no_repite_el_trabajo(self, registry_dir: Path) -> None:
        """Con 'skip' la segunda inserción no escribe nada."""
        registro = ExperimentRegistry(registry_dir)
        registro.append_experiment(fila_experimento())
        assert registro.append_experiment(fila_experimento(notes="segunda")) is False
        filas = registro.read_rows("experiments")
        assert len(filas) == 1
        assert filas[0]["notes"] == ""

    def test_overwrite_sustituye(self, registry_dir: Path) -> None:
        """Con 'overwrite' queda una sola fila, la nueva."""
        registro = ExperimentRegistry(registry_dir)
        registro.append_experiment(fila_experimento())
        registro.append_experiment(fila_experimento(notes="segunda"), on_duplicate="overwrite")
        filas = registro.read_rows("experiments")
        assert len(filas) == 1
        assert filas[0]["notes"] == "segunda"

    def test_version_conserva_ambas(self, registry_dir: Path) -> None:
        """Con 'version' no se pierde ninguna de las dos filas."""
        registro = ExperimentRegistry(registry_dir)
        registro.append_experiment(fila_experimento())
        registro.append_experiment(fila_experimento(notes="segunda"), on_duplicate="version")
        assert len(registro.read_rows("experiments")) == 2

    def test_fail_lanza_excepcion(self, registry_dir: Path) -> None:
        """Con 'fail' un duplicado es un error."""
        registro = ExperimentRegistry(registry_dir)
        registro.append_experiment(fila_experimento())
        with pytest.raises(RegistryDuplicateError):
            registro.append_experiment(fila_experimento(), on_duplicate="fail")

    def test_politica_desconocida(self, registry_dir: Path) -> None:
        """Una política inventada falla antes de tocar el registro."""
        registro = ExperimentRegistry(registry_dir)
        with pytest.raises(ValueError, match="Política de duplicados"):
            registro.append_experiment(fila_experimento(), on_duplicate="lo_que_sea")

    def test_la_clave_de_despliegue_incluye_el_lote(self, registry_dir: Path) -> None:
        """Dos lotes distintos del mismo modelo son dos medidas distintas."""
        registro = ExperimentRegistry(registry_dir)
        registro.append_deployment_metrics(fila_despliegue(batch_size=1))
        registro.append_deployment_metrics(fila_despliegue(batch_size=8))
        assert len(registro.read_rows("deployment_metrics")) == 2

    def test_la_clave_de_despliegue_incluye_el_dispositivo(self, registry_dir: Path) -> None:
        """El mismo modelo medido en dos dispositivos son dos filas."""
        registro = ExperimentRegistry(registry_dir)
        registro.append_deployment_metrics(fila_despliegue(device_id="workstation_cpu"))
        registro.append_deployment_metrics(fila_despliegue(device_id="workstation_gpu"))
        assert len(registro.read_rows("deployment_metrics")) == 2


class TestEscrituraAtomica:
    """Robustez de la escritura frente a interrupciones."""

    def test_no_deja_ficheros_temporales(self, registry_dir: Path) -> None:
        """Tras una escritura correcta no queda ningún fichero temporal."""
        registro = ExperimentRegistry(registry_dir)
        registro.append_experiment(fila_experimento())
        assert list(registry_dir.glob("*.tmp")) == []

    def test_una_escritura_fallida_no_corrompe_el_registro(
        self, registry_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Si el renombrado falla, el registro anterior sigue intacto y legible."""
        registro = ExperimentRegistry(registry_dir)
        registro.append_experiment(fila_experimento("s_baseline_default_a1b2c3d4"))
        contenido_previo = (registry_dir / "experiments.csv").read_text(encoding="utf-8")

        def _fallar(*args, **kwargs):
            raise OSError("fallo simulado durante el renombrado")

        monkeypatch.setattr("tfm_edge.common.registry.os.replace", _fallar)
        with pytest.raises(OSError):
            registro.append_experiment(fila_experimento("s_baseline_otra_b2c3d4e5"))

        assert (registry_dir / "experiments.csv").read_text(encoding="utf-8") == contenido_previo
        assert list(registry_dir.glob("*.tmp")) == []


# ---------------------------------------------------------------------------
# Integridad (sección 16.6)
# ---------------------------------------------------------------------------


class TestIntegridad:
    """Reglas de integridad referencial y filtrado por estado."""

    def test_registro_coherente_valida(self, registry_dir: Path) -> None:
        """Un registro bien formado no produce errores."""
        registro = ExperimentRegistry(registry_dir)
        registro.append_experiment(fila_experimento())
        registro.append_deployment_metrics(fila_despliegue(environment_id=None))
        informe = registro.validate()
        assert informe.is_valid, informe.summary()

    def test_detecta_experiment_id_huerfano(self, registry_dir: Path) -> None:
        """Una medida sin su experimento en experiments.csv es un error."""
        registro = ExperimentRegistry(registry_dir)
        registro.append_deployment_metrics(fila_despliegue("s_pruning_fantasma_ffffffff"))
        informe = registro.validate()
        assert not informe.is_valid
        assert any("huérfano" in error for error in informe.errors)
        with pytest.raises(RegistryIntegrityError):
            informe.raise_if_invalid()

    def test_detecta_environment_id_huerfano(self, registry_dir: Path) -> None:
        """Un entorno referenciado que no existe es un error."""
        registro = ExperimentRegistry(registry_dir)
        registro.append_experiment(fila_experimento())
        registro.append_deployment_metrics(fila_despliegue(environment_id="entorno_inexistente"))
        informe = registro.validate()
        assert any("environment_id" in error for error in informe.errors)

    def test_el_analisis_descarta_las_filas_no_ok_y_lo_reporta(self, registry_dir: Path) -> None:
        """Las filas fallidas o no disponibles se excluyen y se cuentan."""
        registro = ExperimentRegistry(registry_dir)
        registro.append_experiment(fila_experimento("s_baseline_ok_a1b2c3d4", status="ok"))
        registro.append_experiment(fila_experimento("s_baseline_malo_b1b2c3d4", status="fallido"))
        registro.append_experiment(
            fila_experimento("s_baseline_sinrt_c1b2c3d4", status="no_disponible")
        )
        validas, descartadas = registro.valid_rows("experiments")
        assert len(validas) == 1
        assert descartadas == 2

    def test_recuento_de_estados_en_el_informe(self, registry_dir: Path) -> None:
        """El informe de validación resume cuántas filas hay de cada estado."""
        registro = ExperimentRegistry(registry_dir)
        registro.append_experiment(fila_experimento("s_baseline_ok_a1b2c3d4"))
        registro.append_experiment(fila_experimento("s_baseline_malo_b1b2c3d4", status="fallido"))
        informe = registro.validate()
        assert informe.status_counts["experiments"] == {"ok": 1, "fallido": 1}
        assert "experiments: 2 filas" in informe.summary()

    def test_registro_vacio_es_valido(self, registry_dir: Path) -> None:
        """Un registro sin ficheros todavía no viola ninguna regla."""
        assert ExperimentRegistry(registry_dir).validate().is_valid


class TestInicializacion:
    """Creación de las tablas vacías."""

    def test_crea_las_tablas_que_puede(self, registry_dir: Path) -> None:
        """Sin nombres de clase se crean tres de las cuatro tablas."""
        creados = ExperimentRegistry(registry_dir).initialize_empty()
        nombres = {ruta.name for ruta in creados}
        assert nombres == {"experiments.csv", "deployment_metrics.csv", "environments.csv"}

    def test_crea_las_cuatro_con_nombres_de_clase(
        self, registry_dir: Path, class_names: list[str]
    ) -> None:
        """Con los nombres de clase se pueden crear las cuatro tablas."""
        creados = ExperimentRegistry(registry_dir, class_names=class_names).initialize_empty()
        assert len(creados) == 4

    def test_es_idempotente(self, registry_dir: Path) -> None:
        """Inicializar dos veces no vuelve a crear ni sobrescribe nada."""
        registro = ExperimentRegistry(registry_dir)
        registro.initialize_empty()
        assert registro.initialize_empty() == []


# ---------------------------------------------------------------------------
# Huella de entorno
# ---------------------------------------------------------------------------


class TestHuellaDeEntorno:
    """Identificador estable del entorno."""

    def test_el_identificador_ignora_el_timestamp(self) -> None:
        """Dos capturas del mismo entorno en instantes distintos son el mismo entorno."""
        base = {columna: "" for columna in ENVIRONMENT_COLUMNS}
        base.update({"device_id": "workstation_cpu", "os": "Linux", "cpu_model": "CPU X"})
        primera = dict(base, timestamp="2026-01-01T00:00:00+00:00")
        segunda = dict(base, timestamp="2026-08-07T12:00:00+00:00")
        assert compute_environment_id(primera) == compute_environment_id(segunda)

    def test_el_identificador_cambia_con_el_dispositivo(self) -> None:
        """La misma máquina forzada a CPU es otro entorno de medida."""
        base = {columna: "" for columna in ENVIRONMENT_COLUMNS}
        en_gpu = dict(base, device_id="workstation_gpu")
        en_cpu = dict(base, device_id="workstation_cpu")
        assert compute_environment_id(en_gpu) != compute_environment_id(en_cpu)

    def test_el_identificador_cambia_con_la_version_de_una_biblioteca(self) -> None:
        """Cambiar de versión de PyTorch cambia el entorno de medida."""
        base = {columna: "" for columna in ENVIRONMENT_COLUMNS}
        assert compute_environment_id(dict(base, torch_version="2.8.0")) != compute_environment_id(
            dict(base, torch_version="2.9.0")
        )


# Rutas del registro.


class TestRutasRelativas:
    """Las columnas de ruta no pueden atar el registro a la máquina que lo escribió."""

    def test_una_ruta_absoluta_del_arbol_se_guarda_relativa(self, tmp_path: Path) -> None:
        registro = ExperimentRegistry(tmp_path)
        absoluta = get_project_paths().root / "models" / "baselines" / "s_baseline_default"
        registro.append_experiment(
            fila_experimento(
                config_path=str(absoluta / "args.yaml"),
                weights_path=str(absoluta / "best.pt"),
            ),
            on_duplicate="overwrite",
        )
        fila = registro.read_rows("experiments")[0]
        assert fila["config_path"] == "models/baselines/s_baseline_default/args.yaml"
        assert fila["weights_path"] == "models/baselines/s_baseline_default/best.pt"

    def test_una_celda_vacia_sigue_vacia(self, tmp_path: Path) -> None:
        registro = ExperimentRegistry(tmp_path)
        registro.append_experiment(
            fila_experimento(config_path="", weights_path=None), on_duplicate="overwrite"
        )
        fila = registro.read_rows("experiments")[0]
        assert fila["config_path"] == ""
        assert fila["weights_path"] == ""

    def test_una_ruta_de_fuera_del_arbol_no_se_falsea(self, tmp_path: Path) -> None:
        """La de Colab se queda absoluta, y la validación la señala."""
        registro = ExperimentRegistry(tmp_path)
        registro.append_experiment(
            fila_experimento(weights_path="/content/drive/MyDrive/tfm-edge/runs/best.pt"),
            on_duplicate="overwrite",
        )
        fila = registro.read_rows("experiments")[0]
        assert fila["weights_path"] == "/content/drive/MyDrive/tfm-edge/runs/best.pt"

        informe = registro.validate()
        assert any("ruta absoluta" in aviso for aviso in informe.warnings)
        assert informe.is_valid, "una ruta sin reapuntar no invalida la medición"

    def test_tambien_alcanza_a_las_otras_tablas(self, tmp_path: Path) -> None:
        registro = ExperimentRegistry(tmp_path)
        raiz = get_project_paths().root
        registro.append_experiment(fila_experimento(), on_duplicate="overwrite")
        registro.append_deployment_metrics(
            fila_despliegue(
                artifact_path=str(raiz / "results" / "exports" / "modelo.onnx"),
                raw_timings_path=str(raiz / "results" / "raw" / "benchmark" / "t.json"),
            ),
            on_duplicate="overwrite",
        )
        fila = registro.read_rows("deployment_metrics")[0]
        assert fila["artifact_path"] == "results/exports/modelo.onnx"
        assert fila["raw_timings_path"] == "results/raw/benchmark/t.json"


class TestDeteccionDeRutaAbsoluta:
    """``is_absolute_path_text`` reconoce las tres formas que circulan por el proyecto."""

    @pytest.mark.parametrize(
        "texto",
        [
            "/content/drive/MyDrive/tfm-edge/runs/best.pt",
            "/home/usuario/proyecto/models/best.pt",
            "C:\\Users\\usuario\\models\\best.pt",
            "C:/Users/usuario/models/best.pt",
            "\\\\wsl.localhost\\ubuntu\\home\\usuario\\best.pt",
        ],
    )
    def test_absolutas(self, texto: str) -> None:
        """POSIX, unidad de Windows y UNC, corra el test donde corra."""
        assert is_absolute_path_text(texto)

    @pytest.mark.parametrize(
        "texto",
        ["models/baselines/s_baseline_default/best.pt", "results/raw/x.json", "", "   "],
    )
    def test_relativas_y_vacias(self, texto: str) -> None:
        assert not is_absolute_path_text(texto)
