# RuCoCo-2023
Соревнование по разрешению кореферентности для русского языка ([CodaLab](https://codalab.lisn.upsaclay.fr/competitions/9669))


## Описание
Мы приглашаем вас поучаствовать в соревновании по поиску кореферентных выражений в новостных текстах на русском языке. Кореферентными считаются языковые выражения, соответствующие одному объекту действительности, например:
> Соседи [Абрамовича]<sub>0</sub> по поместью недовольны дизайном [его]<sub>0</sub> владения.

## Мотивация
Разрешение кореференции важно для многих задач NLP более высокого уровня, таких как саммаризация, question answering и извлечение информации.
Это уже третье соревнование по разрешению кореференции в рамках конференции Диалог, но первое с таким объемом данных: train/dev датасет содержит около ~1 млн слов.

## Формат данных
Корпус представляет собой файлы в формате JSON для каждого новостного текста в отдельности, разметка - пары начала и конца упоминаний (в символах с начала текста), каждая цепочка в отдельном списке.
```
{
"entities": [[[0, 24], [152, 155]], [[60, 68], [70, 73]]],
"includes": [[], []],
"text": "Лидеры европейских стран собрались на неформальный саммит в Брюсселе, где должны выработать единую позицию по иракской проблеме. По итогам понедельника они заявили, что считают...\n"
}
```
`[[0, 24], [152, 155]]` относится к спанам <strong>Лидеры европейских стран</strong><sub>0</sub> и <strong>они</strong><sub>0</sub> <br />
`[[60, 68], [70, 73]]` к <strong>Брюсселе</strong><sub>1</sub>, <strong>где</strong><sub>1</sub>.

Случаи с расщепленными антецедентами, т.е. когда сущности являются частью одной большой "родительской" сущности, мы выделяем отдельно.
Пример такого случая: <strong>Tom</strong> и <strong>Sid</strong> в "родительской" сущности <strong>they</strong> в этом предложении:
> At half-past nine, that night, <u><strong>Tom</strong></u><sub>0</sub><u> and <strong>Sid</strong></u><sub>1</sub> were sent to bed, as usual. <u>They</u><sub>0,2</sub> said their prayers, and <strong>Sid</strong><sub>1</sub> was soon asleep.<br /></p>

В разметке JSON-файлов для обозначения этого явления используется ключ "include":
```
{
"entities" : [[[31, 34]], [[39, 42], [100, 103]], [[71, 75]]],
"includes" : [[], [], [0, 1]],
"text": "Лидеры европейских стран собрались на неформальный саммит в Брюсселе, где должны выработать единую позицию по иракской проблеме. По итогам понедельника они заявили, что считают...\n"
}
```
Где <strong>"includes" : `[[], [], [0, 1]]`</strong> в этом случае обозначает, что сущность #2 (<i>Tom and Sid, they</i>) - это родительская сущность по отношению к сущности 0 (<i>Tom</i>) и 1 (<i>Sid</i>).


## Полезные ссылки
- [Группа в телеграме](https://t.me/rucoco2023)
- [Инструкция для аннотаторов](https://github.com/vdobrovolskii/rucoco/blob/master/coreference_guidelines.md)
- [Статья по корпусу RuCoCo](https://www.dialog-21.ru/media/5756/dobrovolskiivaplusetal072.pdf)

## Оценка решений
Для оценки решений мы используем метрику LEA (a Link-based Entity Aware metric), прочитать про нее можно в оригинальной статье [Moosavi and Strube (2016)](https://aclanthology.org/P16-1060.pdf) или в [статье корпуса RuCoCo](https://www.dialog-21.ru/media/5756/dobrovolskiivaplusetal072.pdf).
Для расщепленных антецедентов и их родительских сущностей скор рассчитывается отдельно, как для еще одной цепочки.

## Базовое решение
Код базового решения находится в папке [baseline](baseline), описание в [статье корпуса RuCoCo](https://www.dialog-21.ru/media/5756/dobrovolskiivaplusetal072.pdf), решение построено с использованием в качестве энкодера модели ruRoberta-large (от Sber AI).

## Таймлайн соревнования:
- 13 января — публикация train и dev датасетов, тестовых данных;
- 13 марта 23:59 (GMT +3) — последний день для отправки решений в фазе public;
- 19 марта 23:59 (GMT +3) — последний день для отправки решений в фазе private;
- 1 апреля — дедлайн для подачи статьи.

## Условия и публикация
Участники, занявшие 1, 2 или 3 место в приватной фазе соревнования, обязаны предоставить docker контейнер с решением, чтобы подтвердить статус победителей. Лидеры лидерборда, не приславшие контейнер по просьбе организаторов, удаляются из финального лидерборда.<br />
Участники соревнования RuCoCo вне зависимости от места в лидерборде могут опубликовать статью с описанием решения и анализом результатов в сборнике [конференции Диалог](https://www.dialog-21.ru) (индексируется SCOPUS). Решение о принятии статьи в сборник SCOPUS принимают рецензенты. Участники, желающие подать статьи в сборник Диалога, должны прислать организаторам docker контейнеры с решением. 

## Организаторы
- Владимир Добровольский (ABBYY)
- Мария Мичурина (РГГУ)
