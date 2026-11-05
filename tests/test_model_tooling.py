"""Tests de la instrumentación de modelos: complejidad, cirugía y calendario."""

from __future__ import annotations

import pytest

from tfm_edge.training import schedule as sched
from tfm_edge.training.schedule import ScheduleError

# ===========================================================================
# Calendario y plan de sesiones (sin dependencias pesadas)
# ===========================================================================


class TestTrainingSchedule:
    """El calendario tiene que ser coherente consigo mismo."""

    def test_el_calendario_por_defecto_es_valido(self) -> None:
        """La receta del estudio se construye sin quejas."""
        calendario = sched.TrainingSchedule()
        assert calendario.epochs == 100
        assert calendario.close_mosaic_epochs < calendario.epochs

    def test_cerrar_el_mosaico_mas_alla_del_final_es_un_error(self) -> None:
        """Un cierre de mosaico >= épocas cambiaría la receta sin avisar."""
        with pytest.raises(ScheduleError, match="close_mosaic_epochs"):
            sched.TrainingSchedule(epochs=5, close_mosaic_epochs=10)

    def test_calentar_mas_alla_del_final_es_un_error(self) -> None:
        """El calentamiento tiene que terminar dentro del entrenamiento."""
        with pytest.raises(ScheduleError, match="warmup_epochs"):
            sched.TrainingSchedule(epochs=2, close_mosaic_epochs=1, warmup_epochs=3.0)

    def test_sin_checkpoints_periodicos_es_un_error(self) -> None:
        """Con sesiones que se cortan, volcar cada época es lo prudente."""
        with pytest.raises(ScheduleError, match="checkpoint_every_epochs"):
            sched.TrainingSchedule(checkpoint_every_epochs=0)

    def test_la_traduccion_a_ultralytics_no_pierde_nada(self) -> None:
        """Todas las claves del calendario llegan a los overrides."""
        overrides = sched.TrainingSchedule(epochs=50, initial_lr=0.005).to_ultralytics_overrides()
        assert overrides["epochs"] == 50
        assert overrides["lr0"] == 0.005
        assert "close_mosaic" in overrides
        assert "warmup_epochs" in overrides


class TestPlanDeSesiones:
    """El plan solo se calcula sobre tiempos medidos."""

    def test_no_se_planifica_sin_medida(self) -> None:
        """Sin tiempo por época medido, falla en lugar de estimar."""
        for valor in (0, -1, None, "", "rapido"):
            with pytest.raises(ScheduleError, match="measured_epoch_minutes"):
                sched.plan_sessions("s", valor)  # type: ignore[arg-type]

    def test_plan_con_los_tiempos_reales_de_la_t4(self) -> None:
        """Con los minutos por época medidos, el plan cuadra con lo observado."""
        plan = sched.plan_sessions("s", 1.942, epochs=100)
        assert plan.total_hours == pytest.approx(3.24, abs=0.01)
        assert plan.fits_in_one_session
        assert plan.epochs_per_session > 100

    def test_el_maestro_ya_no_cabe_bajo_el_margen(self) -> None:
        """yolov8l entró en una sesión, pero por debajo del margen de seguridad."""
        plan = sched.plan_sessions("l", 6.128, epochs=100)
        assert plan.total_hours == pytest.approx(10.21, abs=0.01)
        assert not plan.fits_in_one_session
        assert plan.epochs_per_session == 99
        # Sin margen sí entra: es el margen, y no el límite, lo que lo descarta.
        sin_margen = sched.plan_sessions("l", 6.128, epochs=100, safety_fraction=0.0)
        assert sin_margen.fits_in_one_session

    def test_un_ritmo_inviable_se_declara_inviable(self) -> None:
        with pytest.raises(ScheduleError, match="Ni una sola época"):
            sched.plan_sessions("l", 2000.0, epochs=100)

    def test_una_campana_que_no_cabe_se_marca(self) -> None:
        """Lo que necesita reanudación queda identificado."""
        plan = sched.plan_campaign({"l": 20.0}, epochs=100)
        assert plan["sizes_needing_resume"] == ["l"]
        assert not plan["plans"]["l"]["fits_in_one_session"]

    def test_la_campana_vacia_es_un_error(self) -> None:
        """Planificar sin ninguna medida no tiene sentido."""
        with pytest.raises(ScheduleError):
            sched.plan_campaign({})


# ===========================================================================
# Complejidad y cirugía (necesitan un modelo real)
# ===========================================================================

torch = pytest.importorskip("torch", reason="requiere PyTorch")
pytest.importorskip("ultralytics", reason="requiere Ultralytics")

from tfm_edge.models import surgery  # noqa: E402
from tfm_edge.models.complexity import (  # noqa: E402
    ComplexityError,
    compare_complexity,
    measure_complexity,
)
from tfm_edge.models.detector import Detector  # noqa: E402
from tfm_edge.models.surgery import SurgeryError  # noqa: E402

pytestmark_modelo = pytest.mark.requires_torch


@pytest.fixture(scope="module")
def detector_n() -> Detector:
    """Construye un yolov8n sin pesos, que no requiere red."""
    return Detector.from_architecture("n", num_classes=17, input_resolution=640)


@pytest.mark.requires_torch
class TestComplejidad:
    """Las cifras salen de ejecutar el modelo."""

    def test_las_cifras_coinciden_con_las_publicadas(self, detector_n: Detector) -> None:
        """yolov8n a 640 px debe dar del orden de 3 M parámetros y 8 GFLOPs."""
        c = measure_complexity(detector_n, resolution=640)
        assert 2.8e6 < c.params_total < 3.3e6
        assert 7.5 < c.gflops < 9.0
        assert c.gmacs == pytest.approx(c.gflops / 2, abs=1e-3)

    def test_el_desglose_por_capa_suma_el_total(self, detector_n: Detector) -> None:
        """El reparto por capa no puede contar ninguna operación dos veces."""
        c = measure_complexity(detector_n, resolution=320)
        assert c.flops_by_module, "El desglose no debería estar vacío."
        assert sum(c.flops_by_module.values()) == pytest.approx(c.gflops, rel=1e-4)

    def test_las_operaciones_escalan_con_la_resolucion(self, detector_n: Detector) -> None:
        """Doblar el lado cuadruplica las operaciones, más o menos."""
        baja = measure_complexity(detector_n, resolution=320)
        alta = measure_complexity(detector_n, resolution=640)
        assert alta.gflops / baja.gflops == pytest.approx(4.0, rel=0.05)
        assert alta.params_total == baja.params_total

    def test_una_resolucion_no_multiplo_de_32_se_rechaza(self, detector_n: Detector) -> None:
        with pytest.raises(ComplexityError, match="múltiplo de 32"):
            measure_complexity(detector_n, resolution=500)

    def test_no_se_comparan_medidas_de_resoluciones_distintas(self, detector_n: Detector) -> None:
        """Comparar 320 con 640 daría una 'reducción' que no existe."""
        baja = measure_complexity(detector_n, resolution=320)
        alta = measure_complexity(detector_n, resolution=640)
        with pytest.raises(ComplexityError, match="no son comparables"):
            compare_complexity(alta, baja)

    def test_una_compresion_que_no_comprime_se_denuncia(self, detector_n: Detector) -> None:
        """Comparar un modelo consigo mismo no es una compresión estructural."""
        c = measure_complexity(detector_n, resolution=320)
        resultado = compare_complexity(c, c)
        assert not resultado["structural"]
        assert "NO es estructural" in resultado["notes"]


@pytest.mark.requires_torch
class TestCirugia:
    """Toda intervención comprueba que ha surtido efecto."""

    def test_la_cabeza_esta_protegida(self, detector_n: Detector) -> None:
        """La cabeza de detección no entra en las candidatas a poda."""
        protegidas = surgery.protected_layer_paths(detector_n)
        assert protegidas, "La lista de capas protegidas no puede estar vacía."
        podables = {n for n, _ in surgery.iter_prunable_convs(detector_n)}
        assert not (podables & protegidas)

    def test_intervenir_una_capa_protegida_se_rechaza(self, detector_n: Detector) -> None:
        """Podar la cabeza cambiaría el número de clases predichas."""
        protegida = sorted(surgery.protected_layer_paths(detector_n))[0]
        with pytest.raises(SurgeryError, match="protegidas"):
            surgery.assert_not_protected(detector_n, [protegida])

    def test_una_ruta_inexistente_dice_donde_falla(self, detector_n: Detector) -> None:
        with pytest.raises(SurgeryError, match="model.99"):
            surgery.get_submodule(detector_n.model, "model.99.noexiste")

    def test_la_sustitucion_se_verifica(self, detector_n: Detector) -> None:
        """replace_submodule comprueba que el módulo nuevo está de verdad."""
        from torch import nn

        nombre, conv = next(surgery.iter_prunable_convs(detector_n))
        nueva = nn.Conv2d(conv.in_channels, conv.out_channels, 3, padding=1, bias=False)
        anterior = surgery.replace_submodule(detector_n.model, nombre, nueva)
        try:
            assert surgery.get_submodule(detector_n.model, nombre) is nueva
        finally:
            surgery.replace_submodule(detector_n.model, nombre, anterior)
        assert surgery.get_submodule(detector_n.model, nombre) is conv

    def test_la_huella_detecta_un_cambio_de_canales(self, detector_n: Detector) -> None:
        """La huella es lo que permite comprobar que una poda fue real."""
        from torch import nn

        antes = surgery.structural_fingerprint(detector_n)
        nombre, conv = next(surgery.iter_prunable_convs(detector_n))
        recortada = nn.Conv2d(
            conv.in_channels, max(1, conv.out_channels // 2), 3, padding=1, bias=False
        )
        anterior = surgery.replace_submodule(detector_n.model, nombre, recortada)
        try:
            despues = surgery.structural_fingerprint(detector_n)
            diferencias = antes.diff(despues)
            assert not diferencias["identical"]
            assert nombre in diferencias["convs_reshaped"]
            assert despues.num_parameters < antes.num_parameters
        finally:
            surgery.replace_submodule(detector_n.model, nombre, anterior)

        # Y tras restaurar, la huella vuelve a ser la misma.
        assert antes.diff(surgery.structural_fingerprint(detector_n))["identical"]

    def test_la_validacion_del_paso_exige_lote_de_al_menos_dos(self, detector_n: Detector) -> None:
        """Con lote 1 hay errores de dimensión que no se manifiestan."""
        with pytest.raises(SurgeryError, match="batch_size"):
            surgery.assert_forward_pass_works(detector_n, resolution=320, batch_size=1)

    def test_el_paso_hacia_delante_funciona(self, detector_n: Detector) -> None:
        """La validación obligatoria tras una poda pasa sobre el modelo intacto."""
        resultado = surgery.assert_forward_pass_works(detector_n, resolution=320)
        assert resultado["ok"]
        assert resultado["input_shape"] == (2, 3, 320, 320)

    def test_congelar_y_descongelar_son_simetricos(self, detector_n: Detector) -> None:
        """Descongelar deshace lo que congeló."""
        inicial = surgery.frozen_summary(detector_n.model)
        surgery.freeze_parameters(detector_n.model, ["model.0.", "model.1."])
        congelado = surgery.frozen_summary(detector_n.model)
        assert congelado["frozen"] > inicial["frozen"]
        surgery.unfreeze_all(detector_n.model)
        assert surgery.frozen_summary(detector_n.model)["frozen"] == 0

    def test_descongelar_devuelve_lo_que_estaba_congelado(self, detector_n: Detector) -> None:
        """Lo devuelto son los elementos reactivados, y nada más."""
        surgery.unfreeze_all(detector_n.model)
        surgery.freeze_parameters(detector_n.model, ["model.0.", "model.1."])
        congelados = surgery.frozen_summary(detector_n.model)["frozen"]
        assert surgery.unfreeze_all(detector_n.model) == congelados
        assert surgery.unfreeze_all(detector_n.model) == 0

    def test_descongelar_no_revienta_con_parametros_enteros(self) -> None:
        """Un parámetro entero no puede llevar gradiente, y pedirlo es un error."""
        import torch
        from torch import nn

        modulo = nn.Module()
        modulo.flotante = nn.Parameter(torch.zeros(4), requires_grad=False)
        modulo.entero = nn.Parameter(torch.zeros(4, dtype=torch.int8), requires_grad=False)

        assert surgery.unfreeze_all(modulo) == 4

        assert modulo.flotante.requires_grad
        assert not modulo.entero.requires_grad
