# Cosas probadas

## Cambios en el modelo

Cualquier cambio en el modelo afecta a la salida, incluso añadir una capa como 

```python
self.skip_fuse = nn.Conv3d(in_channels * 2, in_channels, kernel_size=1)
```

aunque no se use, afecta a la inicialización aleatoria, también a algunas cuestiones relacionadas
con el BatchNormalization, y afecta a la salida

```
ANTES:
NO: Model validation loss: 0.09642401337623596, MSE: 2.2978854179382324, Kriging validation loss: 0.24331200122833252, Kriging MSE: 9.192278861999512
NO: Model validation loss: 0.09642401337623596, MSE: 2.2978854179382324 IDW validation loss: 0.2424629032611847, IDW MSE: 10.048871994018555
NO2: Model validation loss: 0.11129584163427353, MSE: 2.3187599182128906, Kriging validation loss: 0.2623406648635864, Kriging MSE: 10.01021957397461
NO2: Model validation loss: 0.11129584163427353, MSE: 2.3187599182128906 IDW validation loss: 0.2711605131626129, IDW MSE: 10.91378402709961
NOX: Model validation loss: 0.24450747668743134, MSE: 13.592843055725098, Kriging validation loss: 0.6349244117736816, Kriging MSE: 60.46812438964844
NOX: Model validation loss: 0.24450747668743134, MSE: 13.592843055725098 IDW validation loss: 0.6422809958457947, IDW MSE: 65.48436737060547

DESPUES:
NO: Model validation loss: 0.12724988162517548, MSE: 3.774080991744995, Kriging validation loss: 0.24331200122833252, Kriging MSE: 9.192278861999512
NO: Model validation loss: 0.12724988162517548, MSE: 3.774080991744995 IDW validation loss: 0.2424629032611847, IDW MSE: 10.048871994018555
NO2: Model validation loss: 0.09483376145362854, MSE: 1.9931739568710327, Kriging validation loss: 0.2623406648635864, Kriging MSE: 10.01021957397461
NO2: Model validation loss: 0.09483376145362854, MSE: 1.9931739568710327 IDW validation loss: 0.2711605131626129, IDW MSE: 10.91378402709961
NOX: Model validation loss: 0.2811107635498047, MSE: 18.476152420043945, Kriging validation loss: 0.6349244117736816, Kriging MSE: 60.46812438964844
NOX: Model validation loss: 0.2811107635498047, MSE: 18.476152420043945 IDW validation loss: 0.6422809958457947, IDW MSE: 65.48436737060547
```

si sin embargo uso la capa aprendida, entoces es peor...

```
NO: Model validation loss: 0.17194202542304993, MSE: 5.999051094055176, Kriging validation loss: 0.24331200122833252, Kriging MSE: 9.192278861999512
NO: Model validation loss: 0.17194202542304993, MSE: 5.999051094055176 IDW validation loss: 0.2424629032611847, IDW MSE: 10.048871994018555
NO2: Model validation loss: 0.20553521811962128, MSE: 6.516449451446533, Kriging validation loss: 0.2623406648635864, Kriging MSE: 10.01021957397461
NO2: Model validation loss: 0.20553521811962128, MSE: 6.516449451446533 IDW validation loss: 0.2711605131626129, IDW MSE: 10.91378402709961
NOX: Model validation loss: 0.47077998518943787, MSE: 38.805599212646484, Kriging validation loss: 0.6349244117736816, Kriging MSE: 60.46812438964844
NOX: Model validation loss: 0.47077998518943787, MSE: 38.805599212646484 IDW validation loss: 0.6422809958457947, IDW MSE: 65.48436737060547
```

# Discusión interesante

Puede esto sustituir al Dropout?
