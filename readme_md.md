# Práctica 1.3: Reducciones Paralelas en GPU con CUDA

![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg?style=flat&logo=nvidia)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg?style=flat&logo=cplusplus)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%20Tesla%20T4-76B900.svg?style=flat&logo=nvidia)

Repositorio oficial para la **Práctica 1.3: Reducciones Paralelas en CUDA**, correspondiente a la asignatura de **Programación en Arquitecturas Paralelas y GPU**. 

Este proyecto implementa operaciones vectoriales elemento a elemento y algoritmos de **reducción paralela en árbol binario** aprovechando la memoria compartida (`__shared__`) de la GPU y barreras de sincronización (`__syncthreads()`).

---

## 👥 Información del Equipo

* **Equipo:** 07
* **Integrantes (Orden alfabético por primer apellido):**
  1. **GÓMEZ ENRÍQUEZ**, María Fernanda
  2. **LOZANO PÉREZ**, José Manuel
  3. **MARTÍNEZ ARIAS**, Eduardo Caleb
* **Identificador de Enclave/PC:** `PC01`

---

## 🛠️ Aspectos Técnicos del Código (`practica_cuda.cu`)

El código fue desarrollado de forma modular orientada a objetos en C++/CUDA a través de la clase `OperacionesArreglos`. Cuenta con dos kernels principales:

1. **`kernelAritmetica`**: Ejecuta operaciones vectoriales directas $C[i] = A[i] \odot B[i]$ para Suma, Resta, Multiplicación y Cuadrado.
2. **`kernelReduccion`**: Implementa una reducción en árbol binario con paso variable (`stride >>= 1`):
   * **Carga Inicial:** Los hilos cargan datos de la memoria global DRAM a memoria compartida `__shared__` (SRAM on-chip) rellenando con valores neutros (`0LL`, `LLONG_MIN`, `LLONG_MAX`).
   * **Reducción Nivel 1 (GPU):** Cada bloque realiza la reducción interna a velocidad de registros reduciendo los elementos en $O(\log_2 \text{THREADS\_PER\_BLOCK})$ pasos.
   * **Sincronización:** Cada paso del árbol está protegido por `__syncthreads()` para evitar condiciones de carrera.
   * **Reducción Nivel 2 (Host):** El hilo `0` de cada bloque escribe el resultado parcial en un arreglo en memoria global `deviceParciales`. El CPU recupera estos resultados y los consolida.

---

## 📊 Escenarios de Ejecución

El código admite dos modos principales de trabajo según la configuración de macros al inicio de `practica_cuda.cu`:

### 1. Modo Demostración / Detallado ($N = 40$)
Permite la inspección detallada de las etiquetas de control emitidas por la CPU (`[HOST]`) y por los hilos individuales en la GPU (`[DEVICE]`).

```cpp
#define N 40                    // Tamaño reducido
#define THREADS_PER_BLOCK 32    // Hilos por bloque
#define MODO_DETALLADO 1        // Activa impresiones printf por hilo en Device
```

* **Malla de CUDA:** $2$ bloques de $32$ hilos.
* **Comportamiento:** Cada hilo imprime su `blockIdx.x`, `threadIdx.x`, `indice global` y valor parcial.

### 2. Modo Prueba Masiva / Alto Rendimiento ($N = 4,000,000$)
Evalúa el rendimiento en tiempo real y la capacidad de aceleración de la GPU sin la saturación del búfer de entrada/salida.

```cpp
#define N 4000000               // 4 Millones de elementos
#define THREADS_PER_BLOCK 256   // Reconfiguración a 256 hilos por bloque
#define MODO_DETALLADO 0        // Desactiva impresiones en Device para evitar cuellos de botella
```

* **Malla de CUDA:** $15,625$ bloques de $256$ hilos ($15,625 \times 256 \ge 4,000,000$).
* **Comportamiento:** Procesamiento masivo silencioso con medición de tiempos de ejecución mediante `cudaEvent_t`.

---

## ⚡ Resultados de Rendimiento en GPU (NVIDIA Tesla T4)

Prueba ejecutada con $N = 4,000,000$ de elementos y datos generados aleatoriamente:

| Operación | Malla (Grid × Block) | Resultado Global Obtenido | Tiempo GPU (ms) |
| :--- | :---: | :---: | :---: |
| **Suma Vectorial ($A + B$)** | $15,625 \times 256$ | Arreglo $C$ (4M elementos) | `3.107 ms` |
| **Reducción SUMATORIA** | $15,625 \times 256$ | `1,999,647,421,159` | **`0.427 ms`** |
| **Reducción PROMEDIO** | $15,625 \times 256$ | `499,911.855290` | **`0.392 ms`** |
| **Reducción MÁXIMO** | $15,625 \times 256$ | `1,000,000` | **`0.470 ms`** |
| **Reducción MÍNIMO** | $15,625 \times 256$ | `1` | **`0.503 ms`** |

---

## 🚀 Compilación y Ejecución

### Requisitos Previos
* Compilador CUDA `nvcc` (NVIDIA CUDA Toolkit).
* GPU con Compute Capability compatible (e.g. Tesla T4, RTX series, GTX 10 series).

### Pasos
1. **Compilar el código fuente:**
   ```bash
   nvcc -O3 practica_cuda.cu -o practica_cuda
   ```

2. **Ejecutar el binario:**
   ```bash
   ./practica_cuda
   ```

3. **Verificar el archivo de log:**
   El programa crea o añade automáticamente la trazabilidad de la sesión en el archivo persistente `log_ejecucion_cuda.txt`:
   ```bash
   cat log_ejecucion_cuda.txt
   ```

---

## 📋 Estructura del Menú Interactivo (12 Opciones)

1. `Crear arreglos` — Reserva memoria en Host (`new`) y Device (`cudaMalloc`).
2. `Sumar arreglos` — Kernel de suma vectorial element-wise.
3. `Restar arreglos` — Kernel de resta vectorial element-wise.
4. `Multiplicar arreglos` — Kernel de multiplicación element-wise.
5. `Calcular cuadrado` — Kernel de potenciación al cuadrado element-wise.
6. `Llenar secuencial` — Asigna valores secuenciales ($1, 2, 3, \dots, N$).
7. `Llenar aleatorio` — Genera datos aleatorios en el rango $[1, 1,000,000]$.
8. `Sumatoria (Reduccion GPU)` — Acumula todos los elementos en árbol binario.
9. `Promedio (Reduccion GPU)` — Calcula la media aritmética basada en la sumatoria.
10. `Maximo (Reduccion GPU)` — Obtiene el elemento de mayor valor en el arreglo.
11. `Minimo (Reduccion GPU)` — Obtiene el elemento de menor valor en el arreglo.
12. `Salir` — Libera memoria Host/Device y finaliza.