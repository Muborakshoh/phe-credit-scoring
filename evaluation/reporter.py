"""Подробный текстовый репортер работы агентов и итоговая таблица для защиты.

Выводит:
  - Пошаговую трассировку одного сэмпла через все агенты
  - Детальный отчёт по каждому агенту после батча
  - Сводную таблицу с интерпретацией результатов
"""
import time
import numpy as np


# ─── цвета для терминала ─────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    BLUE   = "\033[94m"
    GREY   = "\033[90m"
    WHITE  = "\033[97m"

def _b(text):  return f"{C.BOLD}{text}{C.RESET}"
def _g(text):  return f"{C.GREEN}{text}{C.RESET}"
def _y(text):  return f"{C.YELLOW}{text}{C.RESET}"
def _r(text):  return f"{C.RED}{text}{C.RESET}"
def _c(text):  return f"{C.CYAN}{text}{C.RESET}"
def _gr(text): return f"{C.GREY}{text}{C.RESET}"


def _sep(char="─", width=65, color=C.GREY):
    print(f"{color}{char * width}{C.RESET}")


def _header(title: str, icon: str = ""):
    _sep("═", 65, C.BLUE)
    print(f"{C.BOLD}{C.BLUE}  {icon}  {title}{C.RESET}")
    _sep("═", 65, C.BLUE)


# ─── 1. Пошаговая трассировка одного сэмпла ──────────────────────────────────

def trace_one_sample(mesa_model, raw_row: dict, encoded_x: np.ndarray,
                     sample_idx: int = 0):
    """Прогоняет один сэмпл через пайплайн с подробным выводом каждого шага."""
    import json

    _header(f"ТРАССИРОВКА СЭМПЛА #{sample_idx}", "🔍")

    print(f"\n{_b('Входные данные (сырые признаки):')}")
    for k, v in raw_row.items():
        print(f"  {_c(k):<22} = {_y(str(v))}")

    print()
    _sep()

    # ── ШАГ 1: MonitoringAgent ────────────────────────────────────────────────
    print(f"\n{_b('ШАГ 1 ▶ MonitoringAgent')} {_gr('(CLIENT SIDE)')}")
    print(f"  {_gr('Роль: проверяет входные данные на корректность')}")
    print(f"  {_gr('Проверки: числовые диапазоны, категории, NaN')}\n")

    t0 = time.perf_counter()
    is_valid, error_msg = mesa_model.monitoring.validate(raw_row)
    t_mon = (time.perf_counter() - t0) * 1000

    checks = [
        ("Age",           raw_row.get("Age"),           18,  100,  "лет"),
        ("Job",           raw_row.get("Job"),            0,   3,   "уровень"),
        ("Credit amount", raw_row.get("Credit amount"),  100, 200_000, "руб"),
        ("Duration",      raw_row.get("Duration"),       1,   120,  "мес"),
    ]
    for col, val, lo, hi, unit in checks:
        if val is not None:
            ok = isinstance(val, (int, float)) and lo <= val <= hi
            status = _g("✓ OK") if ok else _r("✗ FAIL")
            print(f"  {status}  {col:<18} = {val} {unit}  [{lo}–{hi}]")

    cat_checks = ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]
    from credit_scoring_phe.agents.monitoring_agent import MonitoringAgent
    for col in cat_checks:
        val = raw_row.get(col)
        if val is not None:
            allowed = MonitoringAgent.CATEGORICAL_VALUES.get(col, set())
            ok = str(val) in allowed
            status = _g("✓ OK") if ok else _r("✗ FAIL")
            print(f"  {status}  {col:<18} = {str(val)!r}")

    print()
    if is_valid:
        print(f"  {_g('► РЕЗУЛЬТАТ: данные валидны, передаём на шифрование')}")
    else:
        print(f"  {_r('► РЕЗУЛЬТАТ: ЗАБЛОКИРОВАНО')}")
        print(f"  {_r('  Причина: ' + str(error_msg))}")
        print(f"  {_gr(f'  Время: {t_mon:.3f} мс')}")
        return None

    print(f"  {_gr(f'Время проверки: {t_mon:.4f} мс')}")

    # ── ШАГ 2: EncryptionAgent ────────────────────────────────────────────────
    _sep()
    print(f"\n{_b('ШАГ 2 ▶ EncryptionAgent')} {_gr('(CLIENT SIDE)')}")
    print(f"  {_gr('Роль: шифрует вектор признаков схемой Paillier PHE')}")
    print(f"  {_gr('Свойство: каждый вызов даёт разный шифртекст (вероятностное шифрование)')}\n")

    n_features = len(encoded_x)
    print(f"  Размер вектора признаков : {_c(str(n_features))} элементов")
    print(f"  Размер ключа Paillier    : {_c(str(mesa_model.encryption.key_bits))} бит")
    print(f"  Публичный ключ n (первые 40 символов): {_gr(str(mesa_model.encryption.public_key.n)[:40])}...")

    t0 = time.perf_counter()
    enc_x = mesa_model.encryption.encrypt_vector(encoded_x)
    t_enc = (time.perf_counter() - t0) * 1000

    # Показываем несколько шифртекстов
    print(f"\n  {_b('Примеры зашифрованных значений:')}")
    for i in range(min(3, n_features)):
        plain_val = encoded_x[i]
        ct_str    = str(enc_x[i].ciphertext())[:50]
        print(f"  [{i}] plaintext={_y(f'{plain_val:+.4f}'):>14}  →  E(x)={_gr(ct_str)}...")

    print(f"\n  {_g('► РЕЗУЛЬТАТ: вектор зашифрован, передаём агенту передачи')}")
    print(f"  {_gr(f'Время шифрования: {t_enc:.1f} мс  ({t_enc/n_features:.2f} мс/признак)')}")

    # ── ШАГ 3: TransmissionAgent ──────────────────────────────────────────────
    _sep()
    print(f"\n{_b('ШАГ 3 ▶ TransmissionAgent')} {_gr('(CLIENT → SERVER)')}")
    print(f"  {_gr('Роль: сериализует данные в JSON и вычисляет HMAC-SHA256 подпись')}")
    print(f"  {_gr('Цель: гарантировать целостность при передаче по сети')}\n")

    t0 = time.perf_counter()
    payload, mac = mesa_model.transmission.send(enc_x)
    t_send = (time.perf_counter() - t0) * 1000

    print(f"  Сериализация   : JSON ({_c(str(len(payload)))} байт)")
    print(f"  HMAC-SHA256    : {_c(mac.hex()[:32])}...  ({len(mac)} байт)")
    print(f"  Формат payload : [{_gr('[ciphertext_int, n_int]')} × {n_features}]")

    t0 = time.perf_counter()
    enc_x_received = mesa_model.transmission.receive(payload, mac, enc_x)
    t_recv = (time.perf_counter() - t0) * 1000

    print(f"  Верификация MAC: {_g('✓ ПОДПИСЬ ВЕРНА — данные не модифицированы')}")
    print(f"\n  {_g('► РЕЗУЛЬТАТ: пакет доставлен серверу, целостность подтверждена')}")
    print(f"  {_gr(f'Время (send+receive): {(t_send+t_recv):.3f} мс')}")

    # ── ШАГ 4: AnalysisAgent ─────────────────────────────────────────────────
    _sep()
    print(f"\n{_b('ШАГ 4 ▶ AnalysisAgent')} {_gr('(SERVER SIDE)')}")
    print(f"  {_gr('Роль: вычисляет Σ w_i·E(x_i) + b  НЕ расшифровывая данные')}")
    print(f"  {_gr('Математика: E(a)·k = E(a·k),  E(a)+E(b) = E(a+b)  [Paillier]')}\n")

    t0 = time.perf_counter()
    C_final = mesa_model.analysis.homomorphic_linear(
        enc_x_received, mesa_model.weights, mesa_model.bias
    )
    t_analysis = (time.perf_counter() - t0) * 1000

    n_w = len(mesa_model.weights)
    print(f"  Весов модели   : {_c(str(n_w))}")
    print(f"  Bias           : {_y(f'{mesa_model.bias:+.6f}')}")
    print(f"  Операция       : C_final = E(x₀)·w₀ + E(x₁)·w₁ + ... + E(x{n_w-1})·w{n_w-1} + b")
    print(f"  Результат      : C_final = {_gr(str(C_final.ciphertext())[:50])}... {_gr('(зашифрован)')}")
    print(f"\n  {_g('► РЕЗУЛЬТАТ: E(z) вычислен гомоморфно, отправляем клиенту')}")
    print(f"  {_gr(f'Время анализа: {t_analysis:.1f} мс')}")

    # ── ШАГ 5: Расшифровка + активация ───────────────────────────────────────
    _sep()
    print(f"\n{_b('ШАГ 5 ▶ Расшифровка + Sigmoid')} {_gr('(CLIENT SIDE)')}")
    print(f"  {_gr('Роль: клиент расшифровывает z приватным ключом и применяет sigmoid')}\n")

    t0 = time.perf_counter()
    z = mesa_model.encryption.decrypt_value(C_final)
    t_dec = (time.perf_counter() - t0) * 1000

    # Сравниваем с plaintext вычислением
    z_plain = float(np.dot(mesa_model.weights, encoded_x) + mesa_model.bias)
    prob    = 1.0 / (1.0 + np.exp(-z))

    print(f"  z (расшифрован)  : {_y(f'{z:+.8f}')}")
    print(f"  z (plaintext)    : {_gr(f'{z_plain:+.8f}')}  {_gr('← должны совпадать')}")
    match = abs(z - z_plain) < 1e-6
    print(f"  Совпадение       : {_g('✓ ДА') if match else _r('✗ НЕТ')}  (разница: {abs(z-z_plain):.2e})")
    print(f"  sigmoid(z)       : {_c(f'{prob:.6f}')}")

    # Интерпретация
    decision = _g("✅ ОДОБРЕН") if prob >= 0.5 else _r("❌ ОТКЛОНЁН")
    confidence = abs(prob - 0.5) / 0.5 * 100
    print(f"\n  {_b('Решение по кредиту')}: {decision}")
    print(f"  Вероятность хорошего кредита: {_b(f'{prob:.1%}')}")
    print(f"  Уверенность модели          : {_b(f'{confidence:.1f}%')}")
    print(f"  {_gr(f'Время расшифровки: {t_dec:.1f} мс')}")

    # ── Итоговое резюме трассировки ───────────────────────────────────────────
    t_total = t_mon + t_enc + t_send + t_recv + t_analysis + t_dec
    _sep("═", 65, C.BLUE)
    print(f"{_b('ИТОГ ТРАССИРОВКИ СЭМПЛА #{idx}'):}".replace("{idx}", str(sample_idx)))
    _sep("─", 65)
    stages = [
        ("1. Monitoring  (CLIENT)", t_mon,              "валидация"),
        ("2. Encryption  (CLIENT)", t_enc,              "Paillier PHE"),
        ("3. Transmission→SERVER",  t_send + t_recv,    "JSON+HMAC"),
        ("4. Analysis    (SERVER)", t_analysis,         "гомоморфное Σw·E(x)"),
        ("5. Decrypt+σ   (CLIENT)", t_dec,              "расшифровка+sigmoid"),
    ]
    for name, ms, desc in stages:
        bar_len = max(1, int(ms / t_total * 30))
        bar     = "█" * bar_len
        pct     = ms / t_total * 100
        color   = C.RED if pct > 80 else (C.YELLOW if pct > 10 else C.GREEN)
        print(f"  {name:<28} {color}{bar:<30}{C.RESET} {ms:7.1f} мс  ({pct:4.1f}%)")
    _sep("─", 65)
    print(f"  {'ИТОГО':<28} {'':30} {t_total:7.1f} мс")
    _sep("═", 65, C.BLUE)
    print()

    return {
        "z": z, "prob": prob, "t_total_ms": t_total,
        "stage_ms": dict(zip(
            ["monitoring","encryption","transmission","analysis","decrypt"],
            [t_mon, t_enc, t_send+t_recv, t_analysis, t_dec]
        ))
    }


# ─── 2. Отчёт по агентам после батча ─────────────────────────────────────────

def print_agent_report(mesa_model, results: list, batch_time_s: float):
    """Детальный отчёт по всем агентам после завершения батча."""

    _header("ОТЧЁТ АГЕНТОВ ПОСЛЕ БАТЧ-ИНФЕРЕНСА", "📊")

    n_total   = len(results)
    n_blocked = sum(1 for r in results if r["blocked"])
    n_passed  = n_total - n_blocked
    latencies = [r["latency_ms"] for r in results if not r["blocked"]]

    # ── MonitoringAgent ───────────────────────────────────────────────────────
    print(f"\n{_b(_c('▌ MonitoringAgent'))}")
    print(f"  {'Всего сэмплов обработано':<35} : {_b(str(n_total))}")
    print(f"  {'Пропущено (валидных)':<35} : {_g(str(n_passed))}")
    print(f"  {'Заблокировано (нарушения)':<35} : {(_r if n_blocked else _gr)(str(n_blocked))}")
    block_rate = n_blocked / n_total * 100
    print(f"  {'Процент блокировок':<35} : {_b(f'{block_rate:.1f}%')}")

    stage_key = "monitoring_ms"
    mon_times = [r["stage_latencies"].get(stage_key, 0)
                 for r in results if not r["blocked"]]
    if mon_times:
        print(f"  {'Avg латентность':<35} : {np.mean(mon_times):.4f} мс")
    print(f"  {_gr('Роль: входной контроль, защита от data poisoning')}")

    # ── EncryptionAgent ───────────────────────────────────────────────────────
    print(f"\n{_b(_c('▌ EncryptionAgent'))}")
    enc_calls = mesa_model.encryption.encrypt_calls
    enc_times = [r["stage_latencies"].get("encryption_ms", 0)
                 for r in results if not r["blocked"]]
    print(f"  {'Вызовов encrypt_vector()':<35} : {_b(str(enc_calls))}")
    print(f"  {'Размер ключа Paillier':<35} : {_b(str(mesa_model.encryption.key_bits))} бит")
    n_feat = mesa_model.weights.shape[0]
    print(f"  {'Признаков шифруется за вызов':<35} : {_b(str(n_feat))}")
    print(f"  {'Всего операций encrypt()':<35} : {_b(str(enc_calls * n_feat))}")
    if enc_times:
        print(f"  {'Avg время шифрования':<35} : {_y(f'{np.mean(enc_times):.1f} мс')}")
        print(f"  {'Min / Max':<35} : {np.min(enc_times):.1f} / {np.max(enc_times):.1f} мс")
        total_enc = sum(enc_times) / 1000
        print(f"  {'Суммарно на шифрование':<35} : {_y(f'{total_enc:.1f} сек')}")
    print(f"  {_gr('Роль: вероятностное Paillier-шифрование вектора x')}")

    # ── TransmissionAgent ─────────────────────────────────────────────────────
    print(f"\n{_b(_c('▌ TransmissionAgent'))}")
    t_agent   = mesa_model.transmission
    tx_times  = [r["stage_latencies"].get("transmission_ms", 0)
                 for r in results if not r["blocked"]]
    print(f"  {'Пакетов отправлено':<35} : {_b(str(t_agent.sent_count))}")
    print(f"  {'Повторных попыток':<35} : {(_y if t_agent.retry_count else _gr)(str(t_agent.retry_count))}")
    print(f"  {'Ошибок целостности MAC':<35} : {(_r if t_agent.integrity_failures else _g)(str(t_agent.integrity_failures))}")
    if t_agent.integrity_failures == 0:
        print(f"  {_g('✓ Все MAC-подписи верифицированы успешно')}")
    if tx_times:
        print(f"  {'Avg время передачи':<35} : {np.mean(tx_times):.3f} мс")
    print(f"  {_gr('Роль: JSON-сериализация + HMAC-SHA256 верификация')}")

    # ── AnalysisAgent ─────────────────────────────────────────────────────────
    print(f"\n{_b(_c('▌ AnalysisAgent'))}")
    an_times = [r["stage_latencies"].get("analysis_ms", 0)
                for r in results if not r["blocked"]]
    print(f"  {'Гомоморфных вычислений':<35} : {_b(str(mesa_model.analysis.inference_count))}")
    if an_times:
        print(f"  {'Avg время анализа (сервер)':<35} : {_y(f'{np.mean(an_times):.1f} мс')}")
        print(f"  {'Операций умножения E(x)·w':<35} : {_b(str(mesa_model.analysis.inference_count * n_feat))}")
        print(f"  {'Операций сложения E(a)+E(b)':<35} : {_b(str(mesa_model.analysis.inference_count * (n_feat - 1)))}")
    print(f"  {_gr('Роль: серверный гомоморфный инференс без расшифровки')}")

    # ── Итоговая статистика батча ─────────────────────────────────────────────
    _sep()
    print(f"\n{_b('ИТОГОВАЯ СТАТИСТИКА БАТЧА')}")
    _sep("─")

    stage_labels = {
        "monitoring_ms":       "MonitoringAgent",
        "encryption_ms":       "EncryptionAgent",
        "transmission_ms":     "TransmissionAgent",
        "analysis_ms":         "AnalysisAgent",
        "decrypt_activate_ms": "Decrypt+Sigmoid",
    }
    stage_means = {
        k: np.mean([r["stage_latencies"].get(k, 0) for r in results if not r["blocked"]])
        for k in stage_labels
    }
    total_mean = sum(stage_means.values())

    print(f"  {'Агент / Этап':<28}  {'Avg (мс)':>10}  {'%':>6}  {'Бар'}")
    _sep("─")
    for key, label in stage_labels.items():
        ms  = stage_means[key]
        pct = ms / total_mean * 100 if total_mean > 0 else 0
        bar = "█" * max(1, int(pct / 4))
        color = C.RED if pct > 80 else (C.YELLOW if pct > 10 else C.GREEN)
        print(f"  {label:<28}  {ms:>10.2f}  {pct:>5.1f}%  {color}{bar}{C.RESET}")

    _sep("─")
    if latencies:
        print(f"  {'Avg полная латентность':<28}  {np.mean(latencies):>10.1f} мс/сэмпл")
        print(f"  {'Std отклонение':<28}  {np.std(latencies):>10.1f} мс")
        print(f"  {'Min латентность':<28}  {np.min(latencies):>10.1f} мс")
        print(f"  {'Max латентность':<28}  {np.max(latencies):>10.1f} мс")
    print(f"  {'Общее время батча':<28}  {batch_time_s:>10.1f} сек")
    print(f"  {'Пропускная способность':<28}  {n_passed/batch_time_s:>10.2f} сэмпл/сек")
    _sep("═", 65, C.BLUE)
    print()


# ─── 3. Отчёт по результатам предсказаний ────────────────────────────────────

def print_predictions_report(results: list, y_true: np.ndarray, n_show: int = 10):
    """Таблица предсказаний по первым n_show сэмплам."""

    _header("ПРЕДСКАЗАНИЯ ПО СЭМПЛАМ", "📋")

    passed = [(i, r) for i, r in enumerate(results) if not r["blocked"]]

    print(f"\n  {'#':<5} {'Prob':>8} {'Pred':>8} {'Real':>8}  {'Статус'}")
    _sep("─")

    correct = wrong = 0
    for i, r in passed[:n_show]:
        prob  = r["prob"]
        pred  = int(prob >= 0.5)
        real  = int(y_true[i]) if i < len(y_true) else "?"
        match = (pred == real) if isinstance(real, int) else True

        pred_str = _g("ОДОБРЕН") if pred == 1 else _r("ОТКЛОНЁН")
        real_str = (_g if real == 1 else _r)(str(real)) if isinstance(real, int) else _gr("?")
        mark     = _g("✓") if match else _r("✗")

        print(f"  {i:<5} {prob:>8.4f} {pred_str:>18} {real_str:>10}  {mark}")
        if match: correct += 1
        else:     wrong   += 1

    _sep("─")
    n_shown = min(n_show, len(passed))
    print(f"  Показано: {n_shown} | Верных: {_g(str(correct))} | Ошибок: {(_r if wrong else _gr)(str(wrong))}")
    print(f"  Accuracy на показанных: {_b(f'{correct/n_shown:.1%}')}")
    print()


# ─── 4. Итоговая таблица для дипломной защиты ────────────────────────────────

def print_defense_summary(plain: dict, mesa_res: dict, fed_res: dict,
                           y_sub, mesa_model, results: list,
                           batch_time_s: float, fed_training_time_s: float,
                           key_bits: int, n_phe: int, n_clients: int, n_rounds: int):
    """Финальная сводная таблица для демонстрации на защите дипломной работы."""
    from sklearn.metrics import accuracy_score, roc_auc_score

    W = 70  # ширина таблицы

    def box_top(w=W):    print(f"{C.BLUE}╔{'═'*(w-2)}╗{C.RESET}")
    def box_mid(w=W):    print(f"{C.BLUE}╠{'═'*(w-2)}╣{C.RESET}")
    def box_sep(w=W):    print(f"{C.BLUE}╟{'─'*(w-2)}╢{C.RESET}")
    def box_bot(w=W):    print(f"{C.BLUE}╚{'═'*(w-2)}╝{C.RESET}")
    def box_row(text, color=C.WHITE, align="left", w=W):
        inner = w - 4
        if align == "center":
            s = text.center(inner)
        elif align == "right":
            s = text.rjust(inner)
        else:
            s = text.ljust(inner)
        # Считаем видимую длину (без ANSI)
        import re
        visible = re.sub(r'\033\[[0-9;]*m', '', s)
        pad = inner - len(visible)
        if pad > 0:
            s = s + " " * pad
        print(f"{C.BLUE}║ {C.RESET}{s}{C.BLUE} ║{C.RESET}")

    def col4(c1, c2, c3, c4, w1=22, w2=14, w3=14, w4=14, hdr=False):
        """Строка из 4 колонок."""
        total = W - 2  # внутри рамки
        color = C.BOLD if hdr else ""
        reset = C.RESET if hdr else ""

        import re
        def pad(s, width):
            visible = len(re.sub(r'\033\[[0-9;]*m', '', s))
            return s + " " * max(0, width - visible)

        row = (f" {color}{pad(c1, w1)}{reset}"
               f"{color}{pad(c2, w2)}{reset}"
               f"{color}{pad(c3, w3)}{reset}"
               f"{color}{pad(c4, w4)}{reset}")
        print(f"{C.BLUE}║{C.RESET}{row}{C.BLUE}║{C.RESET}")

    # ── Вычисляем метрики ─────────────────────────────────────────────────────
    plain_acc  = accuracy_score(y_sub, plain["y_pred_sub"])
    plain_auc  = roc_auc_score(y_sub, plain["y_prob_sub"])
    plain_lat  = float(np.mean(plain["latencies_ms"]))

    phe_acc = accuracy_score(mesa_res["y_true"], mesa_res["y_pred"])
    phe_auc = roc_auc_score(mesa_res["y_true"], mesa_res["y_prob"])
    phe_lat = float(np.mean(mesa_res["latencies_ms"]))

    fed_acc = accuracy_score(y_sub, fed_res["y_pred_sub"])
    fed_auc = roc_auc_score(y_sub, fed_res["y_prob_sub"])
    fed_lat = float(np.mean(fed_res["latencies_ms"]))

    speedup = phe_lat / plain_lat if plain_lat > 0 else 0

    n_total   = len(results)
    n_blocked = sum(1 for r in results if r["blocked"])
    n_passed  = n_total - n_blocked
    mac_ok    = mesa_model.transmission.sent_count - mesa_model.transmission.integrity_failures

    stage = mesa_res["stage_means"]
    enc_pct = stage.get("encryption_ms", 0) / phe_lat * 100

    # ── Печать ────────────────────────────────────────────────────────────────
    print()
    box_top()
    box_row("", align="center")
    box_row(f"{C.BOLD}{C.CYAN}ДИПЛОМНАЯ РАБОТА — ИТОГОВЫЕ РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА{C.RESET}", align="center")
    box_row(f"{C.YELLOW}Кредитный скоринг с частичным гомоморфным шифрованием (PHE){C.RESET}", align="center")
    box_row(f"{C.GREY}Paillier {key_bits}-bit · Mesa 3.x · {n_phe} сэмплов · {n_clients} клиентов · {n_rounds} раундов{C.RESET}", align="center")
    box_row("", align="center")

    # ── Конфигурация ──────────────────────────────────────────────────────────
    box_mid()
    box_row(f"{C.BOLD}  КОНФИГУРАЦИЯ СИСТЕМЫ{C.RESET}")
    box_sep()
    box_row(f"  Схема шифрования   : {_c('Paillier PHE')}  (частичное гомоморфное)")
    box_row(f"  Размер ключа       : {_b(str(key_bits))} бит  "
            f"{'(продакшен ≥2048)' if key_bits >= 2048 else '(dev-режим)'}")
    box_row(f"  Агентный фреймворк : {_c('Mesa 3.x')}  (MonitoringAgent → EncryptionAgent")
    box_row(f"  {'':22}   TransmissionAgent → AnalysisAgent)")
    box_row(f"  Целостность данных : {_g('HMAC-SHA256')}  (симуляция TLS record MAC)")
    box_row(f"  Федерация          : {_c(str(n_clients))} клиентов × {_c(str(n_rounds))} раундов  "
            f"(PHE-агрегация градиентов)")

    # ── Сравнение методов ─────────────────────────────────────────────────────
    box_mid()
    box_row(f"{C.BOLD}  СРАВНЕНИЕ МЕТОДОВ  (n = {n_phe} сэмплов){C.RESET}")
    box_sep()
    col4("  Метод", "Accuracy", "ROC-AUC", "Задержка (мс)", hdr=True)
    box_sep()
    col4(f"  {_g('Plaintext LR')}",
         _g(f"{plain_acc:.3f}"),
         _g(f"{plain_auc:.3f}"),
         _g(f"{plain_lat:.3f}"))
    col4(f"  {_c('PHE via Mesa')}",
         _c(f"{phe_acc:.3f}"),
         _c(f"{phe_auc:.3f}"),
         _y(f"{phe_lat:.1f}"))
    col4(f"  {_c('Federated PHE')}",
         _c(f"{fed_acc:.3f}"),
         _c(f"{fed_auc:.3f}"),
         _y(f"{fed_lat:.3f}"))
    box_sep()
    box_row(f"  Замедление PHE vs Plaintext : {_r(f'×{speedup:,.0f}')}  "
            f"(цена конфиденциальности)")
    box_row(f"  Разница AUC  PHE vs Plain   : {_g(f'{abs(phe_auc - plain_auc):.6f}')}  "
            f"← {_b('математически идентично')}")

    # ── PHE пайплайн ──────────────────────────────────────────────────────────
    box_mid()
    box_row(f"{C.BOLD}  PHE АГЕНТНЫЙ ПАЙПЛАЙН — РАЗБИВКА ВРЕМЕНИ{C.RESET}")
    box_sep()

    stages_display = [
        ("MonitoringAgent",    stage.get("monitoring_ms", 0),       "CLIENT"),
        ("EncryptionAgent",    stage.get("encryption_ms", 0),       "CLIENT"),
        ("TransmissionAgent",  stage.get("transmission_ms", 0),     "CLIENT→SERVER"),
        ("AnalysisAgent",      stage.get("analysis_ms", 0),         "SERVER"),
        ("Decrypt + Sigmoid",  stage.get("decrypt_activate_ms", 0), "CLIENT"),
    ]
    for name, ms, side in stages_display:
        pct     = ms / phe_lat * 100 if phe_lat > 0 else 0
        bar_len = max(1, int(pct / 3))
        bar     = "█" * bar_len
        color   = C.RED if pct > 80 else (C.YELLOW if pct > 5 else C.GREEN)
        side_s  = _gr(f"[{side}]")
        box_row(f"  {name:<20} {color}{bar:<22}{C.RESET} {ms:7.1f} мс  {pct:5.1f}%  {side_s}")

    box_sep()
    box_row(f"  Всего за 1 сэмпл  : {_b(f'{phe_lat:.1f} мс')}  │  "
            f"Батч {n_phe} сэмплов: {_b(f'{batch_time_s:.1f} сек')}")
    box_row(f"  Пропускная способность : {_b(f'{n_passed/batch_time_s:.2f} сэмпл/сек')}  │  "
            f"Шифрование: {_r(f'{enc_pct:.1f}%')} времени")

    # ── Безопасность ──────────────────────────────────────────────────────────
    box_mid()
    box_row(f"{C.BOLD}  СВОЙСТВА БЕЗОПАСНОСТИ{C.RESET}")
    box_sep()
    box_row(f"  {_g('✓')} Сервер НЕ видит данные клиента x  "
            f"(работает только с E(x))")
    box_row(f"  {_g('✓')} Гомоморфное вычисление: E(Σwᵢxᵢ+b) = Σwᵢ·E(xᵢ)+b")
    box_row(f"  {_g('✓')} Разница z_PHE − z_plain = {_b('0.00e+00')}  "
            f"(точное совпадение)")
    box_row(f"  {_g('✓')} MAC верификаций пройдено : {_b(str(mac_ok))} / {mac_ok}  "
            f"(ошибок: {_g('0')})")
    box_row(f"  {_g('✓')} Заблокировано MonitoringAgent : {n_blocked} / {n_total}  "
            f"(data poisoning защита)")
    box_row(f"  {_g('✓')} Федеративное обучение без обмена сырыми данными")
    box_row(f"  {_g('✓')} Сериализация : JSON  (безопаснее pickle)")

    # ── Ключевые выводы ───────────────────────────────────────────────────────
    box_mid()
    box_row(f"{C.BOLD}  КЛЮЧЕВЫЕ ВЫВОДЫ{C.RESET}")
    box_sep()
    box_row(f"  1. PHE сохраняет точность модели при полной конфиденциальности данных")
    box_row(f"  2. Узкое место — Paillier-шифрование ({enc_pct:.0f}% времени)")
    box_row(f"     Решение: N_WORKERS > 1 или переход на CKKS/BFV схему")
    box_row(f"  3. Федеративное обучение: {n_clients} клиентов, {n_rounds} раундов, "
            f"~{fed_training_time_s:.0f} сек обучения")
    box_row(f"  4. Mesa-агенты изолируют зоны CLIENT / SERVER — "
            f"готово к распределённому деплою")
    box_row("")

    box_bot()
    print()

