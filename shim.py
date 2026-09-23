"""Сервис распознавания имен, организаций, адресов и дат рождения для контроля персональных данных ИИконы.

1С (КИИ_КонтрольПерсональныхДанных) шлет текст, сервис возвращает найденные значения. Распознает Natasha
(MIT, github.com/natasha/natasha) локально: модели зашиты в пакет, текст из контура не уходит.
Номера (ИНН, СНИЛС, карты, телефоны, почта, паспорт) ищет сама 1С - здесь только то, для чего нужен
разбор языка.

    GET  /health            -> {"ok": true, "service": "iikona-ner-shim", "version": "1.6.0"}
    POST /detect {"text"}   -> {"ok": true, "entities": [{"type", "text", "start", "stop", "form", "gender"}]}

Типы: ФИО, ОРГАНИЗАЦИЯ, АДРЕС, ДАТАРОЖДЕНИЯ. Для ФИО поле form - порядок частей в тексте
(L - фамилия, F - имя, M - отчество, например "LFM" или "FL"), gender - "f", "m" или "".
Смещения start/stop - в символах Python; 1С ищет значение по тексту и на смещения не опирается.

Запуск: uv run python shim.py. Параметры окружения: NER_HOST (по умолчанию 127.0.0.1), NER_PORT (9002),
NER_MAX_CHARS (200000).
"""
import json
import os
import re
import sys
import threading
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

from natasha import (AddrExtractor, DatesExtractor, Doc, MorphVocab, NamesExtractor,  # noqa: E402
                     NewsEmbedding, NewsNERTagger, Segmenter)

ВЕРСИЯ = "1.6.0"
ХОСТ = os.environ.get("NER_HOST", "127.0.0.1")
ПОРТ = int(os.environ.get("NER_PORT", "9002"))
МАКС_СИМВОЛОВ = int(os.environ.get("NER_MAX_CHARS", "200000"))

# Модели грузятся один раз; разбор под замком: объекты Natasha не обещают потокобезопасность.
_ЗАМОК = threading.Lock()
_СЕГМЕНТАТОР = Segmenter()
_СЛОВАРЬ = MorphVocab()
_NER = NewsNERTagger(NewsEmbedding())
_ИМЕНА = NamesExtractor(_СЛОВАРЬ)
_АДРЕСА = AddrExtractor(_СЛОВАРЬ)
_ДАТЫ = DatesExtractor(_СЛОВАРЬ)

# Организация клиента пишется с правовой формой; «Microsoft» или «1С» без нее - вендоры, не ПД.
ПРАВОВЫЕ_ФОРМЫ = re.compile(r"(?<![А-Яа-яЁё])(ООО|ОАО|ЗАО|ПАО|АО|ИП|НКО|АНО|ГУП|МУП|ФГУП|ТОО|НАО)(?![А-Яа-яЁё])")
# Адрес - это улица и дом; город или регион сами по себе персональными данными не являются.
ЗНАЧИМЫЕ_ЧАСТИ_АДРЕСА = {"улица", "проспект", "переулок", "бульвар", "шоссе", "площадь", "проезд", "набережная",
                         "дом", "квартира", "офис", "корпус", "строение"}
КОНТЕКСТ_РОЖДЕНИЯ = re.compile(r"(рожд|д\.\s?р\.|г\.\s?р\.)", re.IGNORECASE)


def _род(факт):
    """Род по отчеству, имени, затем фамилии через морфологию; пусто, если не понять."""
    for часть in (факт.middle, факт.first):
        if not часть:
            continue
        if часть.lower().endswith(("вна", "чна", "шна")):
            return "f"
        if часть.lower().endswith(("вич", "ич")):
            return "m"
        разбор = _СЛОВАРЬ.parse(часть)
        if разбор:
            род = разбор[0].tag.gender
            if род in ("femn", "masc"):
                return "f" if род == "femn" else "m"
    return _род_фамилии(факт.last)


def _род_фамилии(фамилия):
    """Одиночная фамилия в падеже: «Петровой» - только женская, «Иванову» - и та и другая, рода нет."""
    if not фамилия:
        return ""
    роды = {р.tag.gender for р in _СЛОВАРЬ.parse(фамилия) if "Surn" in р.tag}
    if роды == {"femn"}:
        return "f"
    if роды == {"masc"}:
        return "m"
    return ""


def _форма(текст, факт):
    """Порядок частей имени в тексте: L - фамилия, F - имя, M - отчество."""
    позиции = []
    for буква, часть in (("L", факт.last), ("F", факт.first), ("M", факт.middle)):
        if часть:
            место = текст.find(часть)
            позиции.append((место if место >= 0 else 999, буква))
    return "".join(б for _, б in sorted(позиции))


def _границы_имени(значение, факт, начало, конец):
    """Спан сужается до частей имени: в «Передайте Петровой» NER захватывает глагол из начала предложения."""
    места = []
    for часть in (факт.last, факт.first, факт.middle):
        if not часть:
            continue
        место = значение.find(часть)
        if место < 0:
            return начало, конец
        места.append((место, место + len(часть)))
    if not места:
        return начало, конец
    return начало + min(м[0] for м in места), начало + max(м[1] for м in места)


def _люди(текст, документ):
    """Спаны PER, соседние через пробел склеиваются: «Петрова Анна» Natasha отдает двумя кусками."""
    спаны = [с for с in документ.spans if с.type == "PER"]
    # Склеиваются только куски через пробелы: через перевод строки идут разные люди списком.
    склеенные = []
    for с in спаны:
        if склеенные and 0 < с.start - склеенные[-1][1] <= 2 and not текст[склеенные[-1][1]:с.start].strip(" "):
            склеенные[-1][1] = с.stop
            склеенные[-1][2] += 1
        else:
            склеенные.append([с.start, с.stop, 1])
    итог = []
    for начало, конец, кусков in склеенные:
        значение = текст[начало:конец]
        найдено = _ИМЕНА.find(значение)
        факт = найдено.fact if найдено else None
        # Сужается только одиночный кусок NER: в склейке факт описывает одного человека из нескольких,
        # и сужение по нему выбросило бы остальных.
        if факт and кусков == 1:
            начало, конец = _границы_имени(значение, факт, начало, конец)
            значение = текст[начало:конец]
        итог.append({"type": "ФИО", "text": значение, "start": начало, "stop": конец,
                     "form": _форма(значение, факт) if факт else "", "gender": _род(факт) if факт else ""})
    return итог


def _организации(текст, документ):
    итог = []
    for с in документ.spans:
        if с.type != "ORG":
            continue
        начало = с.start
        # Правовая форма перед спаном: «ООО «Ромашка»» Natasha иногда размечает без «ООО».
        перед = текст[max(0, начало - 6):начало]
        форма = ПРАВОВЫЕ_ФОРМЫ.search(перед)
        if форма:
            начало = max(0, начало - 6) + форма.start()
        значение = текст[начало:с.stop]
        if ПРАВОВЫЕ_ФОРМЫ.search(значение):
            итог.append({"type": "ОРГАНИЗАЦИЯ", "text": значение, "start": начало, "stop": с.stop,
                         "form": "", "gender": ""})
    return итог


def _адреса(текст):
    части = [ч for ч in _АДРЕСА(текст) if ч.fact.type]
    группы = []
    for ч in части:
        if группы and ч.start - группы[-1]["stop"] <= 3 and not текст[группы[-1]["stop"]:ч.start].strip(", "):
            группы[-1]["stop"] = ч.stop
            группы[-1]["types"].add(ч.fact.type)
        else:
            группы.append({"start": ч.start, "stop": ч.stop, "types": {ч.fact.type}})
    return [{"type": "АДРЕС", "text": текст[г["start"]:г["stop"]], "start": г["start"], "stop": г["stop"],
             "form": "", "gender": ""}
            for г in группы if г["types"] & ЗНАЧИМЫЕ_ЧАСТИ_АДРЕСА]


def _даты_рождения(текст):
    итог = []
    for м in _ДАТЫ(текст):
        окно = текст[max(0, м.start - 30):м.start]
        if КОНТЕКСТ_РОЖДЕНИЯ.search(окно):
            итог.append({"type": "ДАТАРОЖДЕНИЯ", "text": текст[м.start:м.stop], "start": м.start, "stop": м.stop,
                         "form": "", "gender": ""})
    return итог


def распознать(текст):
    """Все сущности текста, по возрастанию начала."""
    with _ЗАМОК:
        документ = Doc(текст)
        документ.segment(_СЕГМЕНТАТОР)
        документ.tag_ner(_NER)
        сущности = _люди(текст, документ) + _организации(текст, документ) + _адреса(текст) + _даты_рождения(текст)
    return sorted(сущности, key=lambda с: (с["start"], -с["stop"]))


class Обработчик(BaseHTTPRequestHandler):
    server_version = "iikona-ner-shim/" + ВЕРСИЯ
    # Медленный клиент не держит поток вечно: чтение тела и заголовков ограничено по времени.
    timeout = 30

    def _ответ(self, код, данные):
        тело = json.dumps(данные, ensure_ascii=False).encode("utf-8")
        self.send_response(код)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(тело)))
        self.end_headers()
        self.wfile.write(тело)

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            self._ответ(200, {"ok": True, "service": "iikona-ner-shim", "version": ВЕРСИЯ})
        else:
            self._ответ(404, {"ok": False, "error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path != "/detect":
            self._ответ(404, {"ok": False, "error": "not found"})
            return
        try:
            длина = int(self.headers.get("Content-Length", "0"))
            if длина < 0:
                self._ответ(400, {"ok": False, "error": "bad content length"})
                return
            if длина > МАКС_СИМВОЛОВ * 4:
                self._ответ(413, {"ok": False, "error": "text too large"})
                return
            запрос = json.loads(self.rfile.read(длина).decode("utf-8"))
            if not isinstance(запрос, dict) or not isinstance(запрос.get("text"), str):
                self._ответ(400, {"ok": False, "error": "body must be {\"text\": string}"})
                return
            текст = запрос["text"]
            # Хвост не обрезается: успешный ответ должен значить, что проверен весь текст.
            if len(текст) > МАКС_СИМВОЛОВ:
                self._ответ(413, {"ok": False, "error": "text too large"})
                return
            self._ответ(200, {"ok": True, "entities": распознать(текст)})
        except (ValueError, UnicodeDecodeError):
            self._ответ(400, {"ok": False, "error": "bad json"})
        except Exception:  # noqa: BLE001 - клиенту JSON-ошибка вместо обрыва соединения; текст в лог не идет
            self._ответ(500, {"ok": False, "error": "internal error"})

    def log_message(self, формат, *аргументы):
        # Тексты запросов в лог не пишутся: в них персональные данные.
        sys.stderr.write("%s %s\n" % (self.address_string(), self.requestline.split(" ")[0]))


if __name__ == "__main__":
    сервер = ThreadingHTTPServer((ХОСТ, ПОРТ), Обработчик)
    print(f"iikona-ner-shim {ВЕРСИЯ} слушает {ХОСТ}:{ПОРТ}", flush=True)
    сервер.serve_forever()
