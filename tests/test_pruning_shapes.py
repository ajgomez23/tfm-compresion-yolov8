"""Tests de la poda estructurada, con las formas como eje."""

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="requiere PyTorch")
pytest.importorskip("ultralytics", reason="requiere Ultralytics")
pytest.importorskip("torch_pruning", reason="requiere torch-pruning")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from tfm_edge.common.config import PruningConfig  # noqa: E402
from tfm_edge.compression.pruning import dependency  # noqa: E402
from tfm_edge.compression.pruning import importance as importance_module  # noqa: E402
from tfm_edge.compression.pruning import schedule, structured  # noqa: E402

RESOLUCION = 320


@pytest.fixture(scope="module")
def detector():
    """Construye un yolov8n sin pesos preentrenados, con semilla fijada."""
    from tfm_edge.common.seed import set_global_seed
    from tfm_edge.models.detector import Detector

    set_global_seed(42)
    return Detector.from_architecture("n", num_classes=17, input_resolution=RESOLUCION)


@pytest.fixture(scope="module")
def preparado(detector):
    """Devuelve el detector ya transformado para poder podarlo."""
    return dependency.make_prunable(detector)


def _config(**cambios) -> PruningConfig:
    """Construye una configuración de poda con valores cómodos para el test."""
    base = {
        "criterion": "l1",
        "ratio": 0.3,
        "distribution": "global",
        "min_channels": 4,
        "protected_layers": ["model.22"],
    }
    base.update(cambios)
    return PruningConfig(**base)


class TestCirugiaPrevia:
    """La transformación que hace podable a YOLOv8."""

    def test_el_modelo_original_no_es_podable(self, detector) -> None:
        """Sin la cirugía hay C2f con chunk, y el grafo se calcula mal."""
        assert not dependency.is_prunable(detector.model)
        assert dependency.count_c2f_modules(detector.model) == 8

    def test_la_transformacion_es_numericamente_equivalente(self, preparado) -> None:
        _, informe = preparado
        assert informe.verified
        assert informe.max_abs_difference == pytest.approx(0.0, abs=1e-5)
        assert informe.parameters_before == informe.parameters_after

    def test_no_queda_ningun_c2f(self, preparado) -> None:
        """La condición que hace construible el grafo."""
        podable, informe = preparado
        assert dependency.is_prunable(podable.model)
        assert len(informe.converted) == 8

    def test_el_detector_original_queda_intacto(self, detector, preparado) -> None:
        assert dependency.count_c2f_modules(detector.model) == 8

    def test_repetir_la_cirugia_es_un_error_explicito(self, preparado) -> None:
        """Silenciarlo escondería que alguien la está aplicando dos veces."""
        podable, _ = preparado
        with pytest.raises(dependency.GraphSurgeryError, match="ningún módulo C2f"):
            dependency.make_prunable(podable)

    def test_regresion_eps_y_momentum_de_la_normalizacion(self, detector) -> None:
        c2f = detector.model.model[2]
        sustituto = dependency.SplitFreeC2f(c2f)
        for mitad in (sustituto.cv0, sustituto.cv1):
            assert mitad.bn.eps == c2f.cv1.bn.eps, "eps no copiado: la salida cambia"
            assert mitad.bn.momentum == c2f.cv1.bn.momentum, "momentum no copiado"

    def test_la_equivalencia_se_comprueba_de_verdad(self, detector, monkeypatch) -> None:
        import copy as copy_module

        from tfm_edge.models.detector import Detector

        entrenado = Detector(
            copy_module.deepcopy(detector.model),
            size="n",
            num_classes=17,
            input_resolution=RESOLUCION,
        )
        generador = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for modulo in entrenado.model.modules():
                if isinstance(modulo, nn.BatchNorm2d):
                    modulo.running_var.copy_(
                        torch.rand(modulo.running_var.shape, generator=generador) * 0.02 + 1e-3
                    )
                    modulo.running_mean.normal_(generator=generador)

        def _romper(self, original):
            """Copia los pesos olvidando eps, que es el fallo real que hubo."""
            c = self.c
            with torch.no_grad():
                for destino, corte in ((self.cv0, slice(0, c)), (self.cv1, slice(c, None))):
                    destino.conv.weight.copy_(original.cv1.conv.weight[corte])
                    destino.bn.weight.copy_(original.cv1.bn.weight[corte])
                    destino.bn.bias.copy_(original.cv1.bn.bias[corte])
                    destino.bn.running_mean.copy_(original.cv1.bn.running_mean[corte])
                    destino.bn.running_var.copy_(original.cv1.bn.running_var[corte])

        monkeypatch.setattr(dependency.SplitFreeC2f, "_transferir_pesos", _romper)
        with pytest.raises(dependency.GraphSurgeryError, match="NO es equivalente"):
            dependency.make_prunable(entrenado)


class TestGrafoDeDependencias:
    """El grafo es la parte delicada, y su ausencia debe notarse."""

    def test_sin_preparar_el_grafo_se_niega_a_construirse(self, detector) -> None:
        """Antes daba un IndexError ilegible en mitad de la estimación."""
        with pytest.raises(dependency.GraphSurgeryError, match="make_prunable"):
            dependency.build_dependency_graph(detector)

    def test_preparado_el_grafo_se_construye(self, preparado) -> None:
        """La condición de partida de todo lo demás."""
        podable, _ = preparado
        assert dependency.build_dependency_graph(podable) is not None

    @pytest.mark.slow
    def test_regresion_las_18_convoluciones_del_c2f(self, detector, preparado) -> None:
        import torch_pruning as tp

        podable, _ = preparado
        grafo = dependency.build_dependency_graph(podable)
        criterio = tp.importance.GroupMagnitudeImportance(p=1)
        rotas = []
        for nombre, conv in podable.get_named_conv_layers().items():
            if nombre.startswith("model.22"):
                continue
            try:
                criterio(grafo.get_pruning_group(conv, tp.prune_conv_out_channels, idxs=[0]))
            except Exception as excepcion:  # noqa: BLE001 - se quiere el tipo real
                rotas.append((nombre, type(excepcion).__name__))
        assert rotas == [], f"capas que siguen rompiendo la importancia: {rotas}"


class TestCapasProtegidas:
    """Una errata aquí deja desprotegida la capa que se quería proteger."""

    def test_un_patron_que_no_coincide_es_un_error(self, preparado) -> None:
        podable, _ = preparado
        with pytest.raises(dependency.PruningError, match="no coinciden"):
            dependency.resolve_protected_modules(podable, ["model.99"])

    def test_la_cabeza_se_resuelve_entera(self, preparado) -> None:
        """model.22 debe cubrir las 19 convoluciones de la cabeza."""
        podable, _ = preparado
        protegidos = dependency.resolve_protected_modules(podable, ["model.22"])
        assert len(protegidos) == 19

    def test_los_patrones_solapados_no_duplican(self, preparado) -> None:
        """Duplicar módulos confundiría al podador."""
        podable, _ = preparado
        protegidos = dependency.resolve_protected_modules(podable, ["model.22", "model.22.cv2"])
        assert len(protegidos) == len(set(id(m) for m in protegidos))


class TestCriteriosDeImportancia:
    """Incluido el de control, que es el que da sentido a los demás."""

    def test_estan_los_cinco_criterios(self) -> None:
        """Los que declara PruningConfig."""
        assert set(importance_module.CRITERIA) == {"l1", "l2", "bn_gamma", "taylor", "random"}

    def test_un_criterio_inventado_se_rechaza(self) -> None:
        """No se cae de vuelta a l1 en silencio."""
        with pytest.raises(dependency.PruningError, match="desconocido"):
            importance_module.build_importance("norma_infinito")

    def test_taylor_sin_gradientes_falla_con_un_mensaje_util(self, preparado) -> None:
        """torch-pruning daría un AttributeError sobre NoneType, ilegible."""
        podable, _ = preparado
        assert not importance_module.has_gradients(podable.model)
        with pytest.raises(importance_module.MissingGradientsError, match="gradientes"):
            importance_module.assert_gradients_available(podable.model, "taylor")

    def test_los_criterios_de_pesos_no_exigen_gradientes(self, preparado) -> None:
        """l1 y l2 solo miran los pesos."""
        podable, _ = preparado
        for criterio in ("l1", "l2", "bn_gamma", "random"):
            importance_module.assert_gradients_available(podable.model, criterio)

    def test_el_criterio_aleatorio_esta_marcado_como_control(self) -> None:
        """Para que nadie produzca un modelo del estudio con él sin querer."""
        assert importance_module.CRITERION_SPECS["random"].is_control
        assert "CONTROL" in importance_module.describe_criteria()

    def test_el_criterio_aleatorio_exige_semilla(self) -> None:
        with pytest.raises(dependency.PruningError, match="semilla"):
            importance_module.build_importance("random")

    def test_los_demas_criterios_no_necesitan_semilla(self) -> None:
        """La exigencia es solo del control; l1 y compañía son deterministas."""
        for criterio in ("l1", "l2", "bn_gamma"):
            assert importance_module.build_importance(criterio) is not None


class TestSemillaDelControlAleatorio:
    """El control aleatorio no depende del estado previo del RNG global."""

    def _puntuaciones(self, importancia, longitudes=(8, 16, 4)):
        """Sortea importancias para unos grupos simulados."""
        # Un grupo de torch-pruning se indexa como group[0] = (dep, idxs), y es lo único que mira
        # RandomImportance.
        return [importancia([(None, list(range(n)))]) for n in longitudes]

    def test_la_misma_semilla_da_el_mismo_sorteo(self) -> None:
        primera = self._puntuaciones(importance_module.build_importance("random", seed=1234))
        segunda = self._puntuaciones(importance_module.build_importance("random", seed=1234))
        for a, b in zip(primera, segunda):
            assert torch.equal(a, b)

    def test_semillas_distintas_dan_sorteos_distintos(self) -> None:
        """Si no, la semilla no estaría llegando al generador."""
        primera = self._puntuaciones(importance_module.build_importance("random", seed=1234))
        otra = self._puntuaciones(importance_module.build_importance("random", seed=4321))
        assert not all(torch.equal(a, b) for a, b in zip(primera, otra))

    def test_el_sorteo_no_depende_del_rng_global(self) -> None:
        torch.manual_seed(0)
        sin_consumo = self._puntuaciones(importance_module.build_importance("random", seed=99))

        torch.manual_seed(0)
        torch.rand(1000)  # lo que consumirían 30 épocas de entrenamiento, en pequeño
        con_consumo = self._puntuaciones(importance_module.build_importance("random", seed=99))

        for a, b in zip(sin_consumo, con_consumo):
            assert torch.equal(a, b)

    def test_el_sorteo_no_toca_el_rng_global(self) -> None:
        torch.manual_seed(0)
        esperado = torch.rand(5)

        torch.manual_seed(0)
        self._puntuaciones(importance_module.build_importance("random", seed=99))
        obtenido = torch.rand(5)

        assert torch.equal(esperado, obtenido)


class TestDerivacionDeSemillas:
    """La derivación por rama, que es lo que aísla las dos ramas entre sí."""

    def test_es_estable_entre_procesos(self) -> None:
        """No usa hash() de Python, que PYTHONHASHSEED aleatoriza."""
        from tfm_edge.common.seed import derive_seed

        assert derive_seed(42, "pruning", "l1") == derive_seed(42, "pruning", "l1")

    def test_cada_rama_recibe_una_semilla_distinta(self) -> None:
        """Si coincidieran, las dos ramas compartirían estado otra vez."""
        from tfm_edge.common.seed import derive_seed

        assert derive_seed(42, "pruning", "l1") != derive_seed(42, "pruning", "random")

    def test_la_semilla_base_cambia_las_derivadas(self) -> None:
        """Cambiar --seed tiene que cambiar de verdad el experimento."""
        from tfm_edge.common.seed import derive_seed

        assert derive_seed(42, "pruning", "random") != derive_seed(43, "pruning", "random")

    def test_cae_en_el_rango_admitido_por_manual_seed(self) -> None:
        from tfm_edge.common.seed import derive_seed

        for base in (0, 42, 2**31, 2**63 - 1):
            valor = derive_seed(base, "pruning", "random")
            assert 0 <= valor < 2**31


@pytest.mark.slow
class TestPodaReal:
    """La poda reduce canales de verdad, y el resultado se ejecuta."""

    @pytest.fixture(scope="class")
    def podado(self, detector):
        """Poda al 30 % con reparto global."""
        return structured.prune_model(detector, _config())

    def test_reduce_parametros_de_verdad(self, podado) -> None:
        """Si no redujera, sería una poda por máscaras y el estudio no valdría."""
        _, resultado = podado
        assert resultado.parameters_after < resultado.parameters_before
        assert resultado.parameter_reduction > 0.0

    def test_el_ratio_pedido_y_el_medido_son_cosas_distintas(self, podado) -> None:
        """Confundirlos en la memoria sería un error de bulto."""
        _, resultado = podado
        assert resultado.ratio_requested == 0.3
        # Podar una salida arrastra la entrada correspondiente de la capa siguiente, así que la
        # reducción medida es mayor que el ratio pedido.
        assert resultado.parameter_reduction > resultado.ratio_requested

    def test_los_canales_bajan_en_muchas_capas(self, podado) -> None:
        """Una poda que solo tocara una capa no sería estructurada."""
        _, resultado = podado
        assert len(resultado.layers_changed) > 20
        for nombre in resultado.layers_changed:
            assert resultado.channels_after[nombre] < resultado.channels_before[nombre]

    def test_la_cabeza_queda_intacta(self, podado) -> None:
        """Podarla cambiaría el número de salidas del detector."""
        _, resultado = podado
        assert resultado.head_intact
        for nombre, canales in resultado.channels_before.items():
            if nombre.startswith("model.22"):
                assert resultado.channels_after[nombre] == canales

    def test_la_validacion_obligatoria_pasa(self, podado) -> None:
        """El pase hacia delante tras podar, que es lo que detecta los fallos."""
        _, resultado = podado
        assert resultado.validation is not None
        assert resultado.validation.ok
        assert resultado.validation.forward_ok

    def test_la_salida_conserva_la_forma_del_detector(self, podado) -> None:
        """21 canales = 17 clases + 4 de caja."""
        _, resultado = podado
        assert resultado.validation.output_shape[1] == 21
        assert resultado.validation.expected_output_channels == 21

    def test_el_modelo_podado_se_ejecuta(self, podado) -> None:
        """La comprobación de punta a punta, con otra resolución de lote."""
        modelo_podado, _ = podado
        entrada = torch.zeros(2, 3, RESOLUCION, RESOLUCION)
        modelo_podado.model.eval()
        with torch.no_grad():
            salida = modelo_podado.model(entrada)
        assert salida[0].shape[0] == 2
        assert salida[0].shape[1] == 21

    def test_ninguna_capa_baja_del_minimo(self, podado) -> None:
        """min_channels protege de capas degeneradas de uno o dos canales."""
        _, resultado = podado
        assert resultado.validation.layers_below_minimum == []
        assert resultado.validation.min_channels_found >= 4

    def test_sobrevive_a_guardar_y_cargar(self, podado, tmp_path) -> None:
        from tfm_edge.models.detector import Detector

        modelo_podado, resultado = podado
        ruta = modelo_podado.save(tmp_path / "podado.pt")
        recuperado = Detector.load(ruta, input_resolution=RESOLUCION)
        assert recuperado.count_parameters() == resultado.parameters_after
        assert recuperado.channels_per_conv() == modelo_podado.channels_per_conv()


@pytest.mark.slow
class TestRepartos:
    """Los tres repartos del apartado 9.3.3."""

    def test_uniforme_poda_todas_las_capas_parecido(self, detector) -> None:
        """Es el control estructural del reparto global."""
        _, resultado = structured.prune_model(detector, _config(distribution="uniform"))
        assert resultado.parameter_reduction > 0.0
        assert resultado.head_intact

    def test_por_bloques_admite_ratios_distintos(self, detector) -> None:
        """Backbone y cuello pueden adelgazar de forma distinta."""
        config = _config(
            distribution="per_block",
            block_ratios={"backbone": 0.2, "neck": 0.4},
        )
        _, resultado = structured.prune_model(detector, config)
        assert resultado.parameter_reduction > 0.0
        assert resultado.head_intact

    def test_un_bloque_inventado_se_rechaza(self, detector) -> None:
        """Un nombre mal escrito no puede pasar como ratio cero."""
        config = _config(distribution="per_block", block_ratios={"columna": 0.5})
        with pytest.raises(dependency.PruningError, match="Bloque desconocido"):
            structured.prune_model(detector, config)

    def test_la_poda_de_control_usa_el_mismo_reparto(self, detector) -> None:
        """Solo cambia el criterio; si cambiara algo más no sería comparable."""
        config = _config()
        _, resultado = structured.compare_against_random(detector, config, random_seed=7)
        assert resultado.criterion == "random"
        assert resultado.ratio_requested == config.ratio
        assert resultado.distribution == config.distribution
        assert resultado.protected_layers == config.protected_layers

    def test_la_poda_de_control_sin_semilla_no_se_ejecuta(self, detector) -> None:
        with pytest.raises(dependency.PruningError, match="semilla"):
            structured.compare_against_random(detector, _config())

    def test_el_control_da_la_misma_arquitectura_tras_consumir_azar(self, detector) -> None:
        config = _config()
        _, primero = structured.compare_against_random(detector, config, random_seed=7)

        torch.rand(1000)  # otro estado global, como el que dejaría un reajuste
        _, segundo = structured.compare_against_random(detector, config, random_seed=7)

        assert primero.parameters_after == segundo.parameters_after
        assert primero.channels_after == segundo.channels_after


@pytest.mark.slow
class TestPlanificador:
    """Una etapa frente a iterativa."""

    def test_una_etapa_da_un_solo_ciclo(self, detector) -> None:
        """El caso por defecto."""
        _, resultado = schedule.run_pruning_schedule(detector, _config())
        assert resultado.mode == "one_shot"
        assert len(resultado.steps) == 1

    def test_la_iterativa_reajusta_entre_ciclos_y_no_al_final(self, detector) -> None:
        ciclos: list[int] = []

        def reajuste(_detector, ciclo: int) -> float:
            """Reajuste simulado que solo anota en qué ciclo se le llamó."""
            ciclos.append(ciclo)
            return 0.5

        # min_channels=2: a 320 px yolov8n es tan estrecho que tres ciclos al 30 % dejan alguna capa
        # del primer C2f por debajo de 4 canales.
        _, resultado = schedule.run_pruning_schedule(
            detector, _config(iterative_steps=3, min_channels=2), finetune=reajuste
        )
        assert resultado.mode == "iterative"
        assert len(resultado.steps) == 3
        assert ciclos == [1, 2], "el reajuste va ENTRE ciclos, no tras el último"
        assert resultado.num_finetunes == 2

    def test_la_reduccion_crece_ciclo_a_ciclo(self, detector) -> None:
        """Si no creciera, los ciclos posteriores no estarían podando."""
        _, resultado = schedule.run_pruning_schedule(
            detector, _config(iterative_steps=3, min_channels=2)
        )
        reducciones = [p.parameter_reduction for p in resultado.steps]
        assert reducciones == sorted(reducciones)
        assert reducciones[-1] > reducciones[0]

    def test_una_metrica_ausente_queda_vacia_y_no_a_cero(self, detector) -> None:
        """Un cero ahí se leería como 'el modelo no detecta nada'."""

        def sin_metrica(_detector, _ciclo: int) -> None:
            """Reajuste que no mide nada."""
            return None

        _, resultado = schedule.run_pruning_schedule(
            detector, _config(iterative_steps=2), finetune=sin_metrica
        )
        intermedio = resultado.steps[0]
        assert intermedio.finetuned
        assert intermedio.metric_after_finetune == ""
        assert "—" in resultado.summary()

    def test_comparar_campanas_de_tamanos_distintos_avisa(self) -> None:
        """Comparar dos modelos de tamaños distintos no mide la estrategia."""
        una = schedule.ScheduleResult(
            mode="one_shot",
            final=structured.PruningResult(parameters_before=1000, parameters_after=700),
        )
        iterativa = schedule.ScheduleResult(
            mode="iterative",
            final=structured.PruningResult(parameters_before=1000, parameters_after=500),
        )
        informe = schedule.compare_schedules(una, iterativa)
        assert not informe["comparable"]
        assert "warning" in informe


@pytest.mark.slow
class TestGuardianDelMinimo:
    """torch-pruning no garantiza min_channels, así que se comprueba después."""

    def test_sobrepasarse_aborta_con_un_mensaje_util(self, detector) -> None:
        """El mínimo se exige de verdad, y el mensaje dice qué hacer."""
        with pytest.raises(dependency.ShapeValidationError, match="por debajo del mínimo"):
            structured.prune_model(detector, _config(min_channels=32))

    def test_el_modelo_de_partida_sobrevive_al_aborto(self, detector) -> None:
        """Un fallo a mitad de campaña no puede destruir el baseline."""
        parametros = detector.count_parameters()
        with pytest.raises(dependency.ShapeValidationError):
            structured.prune_model(detector, _config(min_channels=32))
        assert detector.count_parameters() == parametros
        assert dependency.count_c2f_modules(detector.model) == 8

    def test_las_capas_estrechas_por_diseno_estan_exentas(self, preparado) -> None:
        """``model.22.dfl.conv`` tiene un canal de salida y nunca se poda."""
        podable, _ = preparado
        assert podable.get_module("model.22.dfl.conv").out_channels == 1
        validacion = dependency.validate_after_pruning(
            podable, min_channels=4, raise_on_failure=False, protected_layers=["model.22"]
        )
        assert validacion.ok
        sin_proteger = dependency.validate_after_pruning(
            podable, min_channels=4, raise_on_failure=False
        )
        assert "model.22.dfl.conv" in sin_proteger.layers_below_minimum


@pytest.mark.slow
class TestValidacionDeFormas:
    """La validación tiene que detectar lo que debe detectar."""

    def test_detecta_una_cabeza_podada(self, preparado) -> None:
        """Es el fallo que la protección de capas existe para evitar."""
        import copy

        from tfm_edge.models.detector import Detector

        podable, _ = preparado
        copia = Detector(
            copy.deepcopy(podable.model),
            size="n",
            num_classes=17,
            input_resolution=RESOLUCION,
        )
        # Se poda la cabeza a mano, que es justo lo que nunca debe pasar.
        cabeza = copia.get_module("model.22.cv3.0.2")
        assert isinstance(cabeza, nn.Conv2d)
        recortada = nn.Conv2d(cabeza.in_channels, 5, kernel_size=1)
        copia.get_module("model.22.cv3.0")[2] = recortada

        validacion = dependency.validate_after_pruning(copia, raise_on_failure=False)
        assert not validacion.ok

    def test_detecta_capas_por_debajo_del_minimo(self, preparado) -> None:
        podable, _ = preparado
        validacion = dependency.validate_after_pruning(
            podable, min_channels=64, raise_on_failure=False
        )
        assert not validacion.ok
        assert validacion.layers_below_minimum
        assert validacion.forward_ok, "el modelo se ejecuta; lo que falla es el mínimo"

    def test_un_modelo_sano_pasa(self, preparado) -> None:
        """El caso base, para que el test anterior signifique algo."""
        podable, _ = preparado
        validacion = dependency.validate_after_pruning(podable, min_channels=1)
        assert validacion.ok
        assert validacion.output_shape[1] == 21
