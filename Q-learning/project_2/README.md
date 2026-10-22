# Q-learning en un laberinto

Interfaz para generar laberintos, definir la recompensa y ver cómo
**Q-learning** tabular aprende a salir. El agente es ε-greedy. Cada paso
aplica la actualización de Watkins:

```
Q(s, a) ← Q(s, a) + α [ r + γ max Q(s′, ·) − Q(s, a) ]
```

Chocar con un muro deja al agente en la misma celda. `Q-learning/project`
no se modifica: allí el algoritmo está sobre un pasillo y una cuadrícula
abierta; aquí el mundo tiene muros.

## Setup

```bash
cd "Q-learning/project_2"
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

En Windows (PowerShell):

```powershell
cd "Q-learning/project_2"
python3 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Luego:

```bash
python app.py
```

Se abre el navegador en `http://127.0.0.1:8765`.

Para comprobar generación, la recompensa y que el laberinto de ejemplo se
resuelve:

```bash
python check.py
```

## La interfaz

**Laberinto.** Filas y columnas de 5 a 25. La semilla reproduce el mismo
dibujo. *Pasillos extra* abre muros de un laberinto perfecto (un solo
camino) para que haya atajos. *Generar* usa esos datos. *Aleatorio* elige
otra semilla y genera. *Aplicar salida y meta* mueve S y G sin cambiar los
muros. Las coordenadas empiezan en 0.

**Guardar** escribe `mazes/<nombre>.yaml` y lo deja en la lista.
**Cargar** lo recupera, incluida la función de recompensa.
**Descargar** entrega ese mismo YAML.

**Recompensa.** Cada transición evalúa la expresión. Las variables son:

| Variable | Valor |
|---|---|
| `step` | 1 en cada transición |
| `goal` | 1 al entrar en la meta, si no 0 |
| `hit_wall` | 1 si el movimiento chocó, si no 0 |
| `progress` | distancia Manhattan anterior menos la nueva |

Solo se permiten números, esas variables y `+ - * / ()`.

Presets:

- Escasa: `10 * goal`
- Coste por paso: `-1 + 10 * goal`
- Penalización de muro: `-1 + 10 * goal - 5 * hit_wall`
- Moldeada: `-0.1 + 10 * goal + progress`

**Entrenamiento.** α, γ, ε, episodios, pasos máximos y semilla del agente.
La curva se actualiza mientras corre. Al terminar, las flechas son la
política greedy (en un empate gana el orden N, E, S, W). *Animar camino*
recorre esa política desde S. Durante el entrenamiento los empates se
rompen al azar, para que una tabla en ceros no camine siempre al norte.

## Archivo

`mazes/demo.yaml` es un laberinto 8×8, semilla 7, sin pasillos extra.
La salida es `[0, 0]` y la meta `[7, 7]`.

```yaml
name: demo
rows: 8
cols: 8
start: [0, 0]
goal: [7, 7]
seed: 7
algorithm: recursive_backtracker
extra_passages: 0
reward: -1 + 10 * goal
passages:
  - [[0, 0], [1, 0]]
```

Cada pasillo es un par de celdas vecinas. Antes de guardar, una búsqueda
en anchura comprueba que la meta es alcanzable.

## Qué esperar

Con el demo, la recompensa *Coste por paso* y los valores iniciales
(α = 0.2, γ = 0.95, ε = 0.25, 500 episodios, 300 pasos, semilla del
agente 0), la política greedy llega a la meta en **18** pasos y el
retorno es **−8** (dieciocho veces −1, más 10 al llegar).

La recompensa escasa tarda más en propagarse. En un laberinto grande
conviene subir episodios, γ y los pasos máximos. Si la política no llega,
las flechas muestran dónde se queda atrapada.
