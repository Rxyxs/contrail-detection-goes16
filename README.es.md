[[English](README.md) ] | [Español ]

# Detección de estelas de condensación en imágenes GOES-16

![CI](https://github.com/Rxyxs/contrail-detection-goes16/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)

Segmentación a nivel de píxel de estelas de condensación de aviones (contrails) en imágenes
infrarrojas del satélite GOES-16, para la competencia de Kaggle
[Google Research - Identify Contrails to Reduce Global Warming](https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming).

Las estelas persistentes atrapan el calor que la Tierra emite de noche y explican cerca del 1%
del calentamiento causado por humanos. Las aerolíneas pueden evitar muchas cambiando levemente
de altitud, pero los modelos físicos que predicen dónde se forman necesitan contrastarse con lo
que realmente ven los satélites. Este proyecto construye ese contraste: un modelo que marca cada
píxel de estela en una escena de 256x256.

**Late submission en Kaggle:** Dice global **0,568** en el público y **0,597** en el privado. El privado queda a 0,002 del 0,595 de validación, así que la estimación de validación se sostuvo.

## Resultados

Todas las corridas se miden con la métrica de la competencia (Dice global: intersección y tamaños
sumados sobre todos los píxeles del conjunto) en la validación completa (1.856 registros),
40 épocas cada una.

| Modelo | Registros de train | Dice global |
|---|---|---|
| U-Net ResNet34, frame etiquetado | 2.447 | 0,508 |
| + dos frames vecinos como canales extra | 2.447 | 0,461 |
| + entrada agrandada 2x dentro de la red | 2.447 | 0,555 |
| + entrada agrandada 2x, más datos | 8.197 | **0,595** |

![Curvas de validación](docs/img/val_curves.png)

Lo que dicen los números:

- **La resolución pesa más que todo lo demás que probé.** Las estelas suelen medir 1 a 3 píxeles
  de ancho a 2 km por píxel. Agrandar la entrada 2x antes del encoder evita que desaparezcan en
  las primeras capas con stride 2; los logits se reducen de vuelta a 256 px para que la pérdida y
  la métrica queden en la grilla de las etiquetas. Mismos datos, mismas épocas: +0,047 de Dice.
- **Más datos sigue rindiendo.** Pasar del 12% al 40% del train sumó otros +0,040, y la curva
  seguía subiendo.
- **Apilar frames como canales empeoró** (-0,047). Los frames vecinos están separados por 10
  minutos y las estelas se desplazan, así que quedan desalineados; las soluciones fuertes de la
  competencia usaban módulos temporales específicos.
- **El mejor umbral está muy por debajo de 0,5** (0,1 en el modelo final, 0,005-0,02 en los
  demás). Con menos de 1% de píxeles positivos las probabilidades quedan mal calibradas, así que
  el corte se ajusta en validación. Una grilla lineal que partía en 0,05 lo escondía al principio;
  ahora la grilla es logarítmica en el extremo bajo.
- **Entrenar mucho sobreajusta.** Una corrida de 100 épocas tocó su máximo en la época 31 (0,509)
  y bajó a 0,476 mientras la pérdida de entrenamiento seguía cayendo.

El score de validación elige tanto el checkpoint como el umbral, así que es levemente optimista.
El score privado de Kaggle es el número independiente, y coincide (0,597).

La submission corre sin internet y en CPU en Kaggle ([`kaggle/submit.py`](kaggle/submit.py)). Dos envíos anteriores en GPU fallaron en la re-ejecución sobre el test oculto con un "system error" de la plataforma aunque el notebook terminaba bien; el mismo notebook en CPU se puntuó sin problemas.

![Ejemplos de predicción](docs/img/examples.png)

El modelo detecta bien las estelas rectas y nítidas. La segunda fila es un error: una estela sobre
un fondo nuboso complejo no se detecta. También se pierden fragmentos cortos (tercera fila).

## Datos

- **OpenContrails** (Ng et al., 2023, [arXiv:2304.02122](https://arxiv.org/abs/2304.02122)),
  el dataset de origen de la competencia, publicado por Google con licencia **CC BY 4.0** en
  `gs://goes_contrails_dataset`. Lo usé para 8.197 registros de train (bloques 0-39 de 100) y la
  validación de 1.856 registros. Descargado en octubre de 2026.
- **Datos de la competencia en Kaggle** (reglas de la competencia; no se redistribuyen):
  metadatos, el test oculto al momento de la submission, y la verificación de que los ids de
  OpenContrails coinciden con los de Kaggle.

Cada registro trae 8 frames separados por 10 minutos, con nueve bandas infrarrojas de 256x256;
el quinto frame es el etiquetado. La verdad terreno es el voto mayoritario de 4 o más etiquetadores.

### Cómo obtener los datos sin bajar 451 GB

El archivo de Kaggle pesa 451 GB y pedirle a la API cada uno de los ~90.000 archivos activa su
límite de peticiones (HTTP 429, cuenta bloqueada por horas). Dos cargadores lo evitan:

- [`remote_zip.py`](src/contrails/remote_zip.py) pide a Kaggle una sola vez la URL firmada del
  archivo, lee el directorio central del zip con peticiones HTTP Range y luego baja solo los
  miembros necesarios (bandas 11, 14, 15 y la máscara), una petición por archivo.
- [`opencontrails.py`](src/contrails/opencontrails.py) lee en streaming los bloques TFRecord de
  OpenContrails desde el bucket público y decodifica los `TensorProto` serializados con un lector
  de protobuf de 40 líneas, sin depender de TensorFlow. Se descartan los registros de train cuyo
  id pertenece a la validación.

Ambos guardan cada registro como el compuesto de falso color "ash" (bandas 15-14, 14-11, 14) en
`uint8`, 8 frames más la máscara, ~1,2 MB en vez de ~43 MB.

## Método

```
Bandas GOES-16 11/14/15 (8 frames) ──► RGB ash uint8 ──► frame etiquetado (3 canales)
        │                                                     │
        │                                     agrandado bilineal x2 (512 px)
        │                                                     │
        │                                   U-Net, encoder ResNet34 (inicio ImageNet)
        │                                                     │
        │                                    logits reducidos a 256 px
        │                                                     │
máscaras humanas (>50% de etiquetadores) ──► pérdida BCE + Dice suave, AdamW, OneCycle
                                                              │
                                     umbral ajustado con Dice global en validación
                                                              │
                                     codificación RLE ──► submission.csv
```

Precisión mixta en una RTX 4060 Ti (8 GB): ~0,5 min por época con 2.447 registros a 256 px,
~2,7 min con 8.197 registros a 512 px.

## Reproducir

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.\venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "src"

# metadatos de Kaggle (requiere ~/.kaggle/kaggle.json y aceptar las reglas)
kaggle competitions download google-research-identify-contrails-reduce-global-warming -f validation_metadata.json -p data/raw

# datos: ~300 GB en streaming, ~10 GB en disco
python -m contrails.opencontrails --split validation --shards (0..99)
python -m contrails.opencontrails --split train --shards (0..39)

# experimentos
python -m contrails.train --name r34_f1_e40 --frames 1 --epochs 40 --train-list splits/train_2447.txt
python -m contrails.train --name r34_f3 --frames 3 --epochs 40 --train-list splits/train_2447.txt
python -m contrails.train --name r34_f1_up2 --upscale 2 --batch 8 --lr 3e-4 --epochs 40 --train-list splits/train_2447.txt
python -m contrails.train --name r34_f1_up2_full --upscale 2 --batch 8 --lr 3e-4 --epochs 40
python -m contrails.evaluate --name r34_f1_up2_full
python -m contrails.figures

pytest -q tests
```

La corrida final se detuvo tras la época 38 de 40 al cerrarse la sesión de la máquina; su mejor
checkpoint (época 23) es el que se evaluó y envió. [`kaggle/submit.py`](kaggle/submit.py) es el
script de inferencia sin internet que corre en Kaggle.

## Estructura

```
src/contrails/
  data.py           falso color ash, cargador de Kaggle sobre el archivo remoto
  remote_zip.py     lector HTTP Range de miembros de un zip
  opencontrails.py  streaming de TFRecord / TensorProto desde GCS
  dataset.py        dataset de PyTorch sobre los registros compactos
  train.py          modelo, pérdida y loop de entrenamiento
  evaluate.py       Dice global sobre una grilla de umbrales
  metrics.py        acumulador de Dice global
  rle.py            codificación run-length de la competencia
  predict.py        generador de la submission
  figures.py        figuras del README
kaggle/submit.py    inferencia sin internet en Kaggle
tests/              RLE, métrica, zip remoto, predicción de punta a punta
```

## Limitaciones

- Usé el 40% del train; la curva de aprendizaje no se había aplanado.
- Un solo encoder y una sola semilla. Diferencias de ~0,01 entre corridas están dentro del ruido.
- Sin test-time augmentation ni ensambles. Los flips están en el código pero sin probar: se
  reporta que las etiquetas están corridas medio píxel respecto de las imágenes, y un flip movería
  ese desfase.

## Licencia

Código: MIT. Datos OpenContrails: CC BY 4.0, Google. Los datos de la competencia de Kaggle no se
redistribuyen.
