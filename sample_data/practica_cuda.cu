#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <cstdlib>
#include <climits>
#include <iomanip>
#include <limits>
#include <new>

using namespace std;

#define N 40                   
#define THREADS_PER_BLOCK 32 
#define MODO_DETALLADO 1       
#define EQUIPO "PC01"

#define CUDA_CHECK(call) do { \
    cudaError_t errorCUDA = (call); \
    if (errorCUDA != cudaSuccess) { \
        cerr << "[HOST] [Equipo: " << EQUIPO << "] Error CUDA en " \
             << #call << ": " << cudaGetErrorString(errorCUDA) << endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// 2=Suma, 3=Resta, 4=Multiplicacion, 5=Cuadrado.
__global__ void kernelAritmetica(const int* A, const int* B,
                                 long long* C, int n, int operacion, int detalle) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long valor = 0;
    if      (operacion == 2) valor = (long long)A[i] + B[i];
    else if (operacion == 3) valor = (long long)A[i] - B[i];
    else if (operacion == 4) valor = (long long)A[i] * B[i];
    else if (operacion == 5) valor = (long long)A[i] * A[i];
    C[i] = valor;
    if (detalle) {
        printf("[DEVICE] [Equipo: %s] [Bloque CUDA: %d] [Hilo CUDA: %d] "
               "[Indice Global: %d] [Operacion: %d] Resultado: %lld\n",
               EQUIPO, blockIdx.x, threadIdx.x, i, operacion, valor);
    }
}

// Modo 0: sumatoria; 1: maximo; 2: minimo.
// Cada bloque crea UN resultado parcial en memoria global.
__global__ void kernelReduccion(const int* datos, long long* parciales,
                                int n, int modo, int detalle) {
    extern __shared__ long long compartida[];
    int tid = threadIdx.x;
    int indice = blockIdx.x * blockDim.x + tid;
    long long neutral = (modo == 1) ? LLONG_MIN : ((modo == 2) ? LLONG_MAX : 0LL);
    long long valor = (indice < n) ? (long long)datos[indice] : neutral;
    compartida[tid] = valor;
    if (detalle && indice < n) {
        printf("[DEVICE] [Equipo: %s] [Bloque CUDA: %d] [Hilo CUDA: %d] "
               "[Indice Global: %d] [Operacion: Reduccion %d - Carga] Valor: %lld\n",
               EQUIPO, blockIdx.x, tid, indice, modo, valor);
    }
    __syncthreads();

    // THREADS_PER_BLOCK debe ser potencia de 2.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            long long otro = compartida[tid + stride];
            if (modo == 0) compartida[tid] += otro;
            else if (modo == 1 && otro > compartida[tid]) compartida[tid] = otro;
            else if (modo == 2 && otro < compartida[tid]) compartida[tid] = otro;
        }
        __syncthreads();
    }

    if (tid == 0) {
        parciales[blockIdx.x] = compartida[0];
        if (detalle) {
            printf("[DEVICE] [Equipo: %s] [Bloque CUDA: %d] [Hilo CUDA: 0] "
                   "[Indice Global: %d] [Operacion: Reduccion %d - Parcial] Resultado: %lld\n",
                   EQUIPO, blockIdx.x, blockIdx.x * blockDim.x, modo, compartida[0]);
        }
    }
}

class OperacionesArreglos {
    int *host_A = nullptr, *host_B = nullptr;
    long long *host_C = nullptr;
    int *device_A = nullptr, *device_B = nullptr;
    long long *device_C = nullptr;
    int tamano;
    bool creados = false, inicializados = false;
    ofstream log;

    void escribir(const string& linea, bool consola = true) {
        if (consola) cout << linea << '\n';
        if (log.is_open()) { log << linea << '\n'; log.flush(); }
    }
    string prefijoHost() const { return "[HOST] [Equipo: " + string(EQUIPO) + "] "; }
    string prefijoDevice(int bloque, int hilo, int indice) const {
        return "[DEVICE] [Equipo: " + string(EQUIPO) + "] [Bloque CUDA: " +
               to_string(bloque) + "] [Hilo CUDA: " + to_string(hilo) +
               "] [Indice Global: " + to_string(indice) + "] ";
    }
    void comprobarListo() const { /* la validacion se hace en los metodos */ }

    void copiarGPU() {
        size_t bytesAB = (size_t)tamano * sizeof(int);
        size_t bytesC = (size_t)tamano * sizeof(long long);
        CUDA_CHECK(cudaMemcpy(device_A, host_A, bytesAB, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_B, host_B, bytesAB, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_C, host_C, bytesC, cudaMemcpyHostToDevice));
        registrar("Datos copiados de Host a Device.");
    }

public:
    explicit OperacionesArreglos(int n) : tamano(n) {
        log.open("log_ejecucion_cuda.txt", ios::app);
        if (!log) cerr << "[HOST] [Equipo: " << EQUIPO << "] No se pudo abrir el log\n";
        escribir(prefijoHost() + "========== NUEVA EJECUCION ==========");
    }
    void registrar(const string& mensaje) { escribir(prefijoHost() + mensaje); }
    void registrarSoloArchivo(const string& linea) { escribir(linea, false); }

    void menu() {
        const char* nombres[] = {
            "Crear arreglos", "Sumar arreglos", "Restar arreglos", "Multiplicar arreglos",
            "Calcular cuadrado", "Llenar secuencial", "Llenar aleatorio",
            "Sumatoria (Reduccion GPU)", "Promedio (Reduccion GPU)",
            "Maximo (Reduccion GPU)", "Minimo (Reduccion GPU)", "Salir"
        };
        registrar("========== MENU CUDA ==========");
        for (int i = 0; i < 12; i++) registrar(to_string(i + 1) + ". " + nombres[i]);
        cout << prefijoHost() << "Seleccione una opcion: " << flush;
    }

    void crearArregloCUDA() {
        if (creados) { registrar("Los arreglos ya existen."); return; }
        host_A = new int[tamano]();
        host_B = new int[tamano]();
        host_C = new long long[tamano]();
        size_t bytesAB = (size_t)tamano * sizeof(int);
        size_t bytesC = (size_t)tamano * sizeof(long long);
        CUDA_CHECK(cudaMalloc((void**)&device_A, bytesAB));
        CUDA_CHECK(cudaMalloc((void**)&device_B, bytesAB));
        CUDA_CHECK(cudaMalloc((void**)&device_C, bytesC));
        CUDA_CHECK(cudaMemset(device_A, 0, bytesAB));
        CUDA_CHECK(cudaMemset(device_B, 0, bytesAB));
        CUDA_CHECK(cudaMemset(device_C, 0, bytesC));
        creados = true;
        registrar("Arreglos A, B y C creados. N=" + to_string(tamano));
        registrar("Memoria Host y Device reservada.");
    }

    void mostrarArreglos() {
        #if MODO_DETALLADO
        if (tamano > 40) { 
            registrar("Impresion detallada desactivada."); 
            return; 
        }
        for (int i = 0; i < tamano; i++)
            registrar("[Indice Global: " + to_string(i) + "] A=" + to_string(host_A[i]) +
                      " B=" + to_string(host_B[i]) + " C=" + to_string(host_C[i]));
        #else
        registrar("Impresion detallada desactivada.");
        #endif
    }

    void llenarSecuencial() {
        if (!creados) { registrar("Primero crea los arreglos."); return; }
        for (int i = 0; i < tamano; i++) {
            // Evita exceder el rango de int con N grande.
            host_A[i] = i + 1; host_B[i] = i + 1; host_C[i] = 0;
        }
        copiarGPU(); inicializados = true;
        registrar("Llenado secuencial completado."); mostrarArreglos();
    }
    void llenarAleatorio() {
        if (!creados) { registrar("Primero crea los arreglos."); return; }
        for (int i = 0; i < tamano; i++) {
            host_A[i] = (rand() % 1000000) + 1;
            host_B[i] = (rand() % 1000000) + 1;
            host_C[i] = 0;
        }
        copiarGPU(); inicializados = true;
        registrar("Llenado aleatorio completado."); mostrarArreglos();
    }

    void ejecutarOperacion(int opcion) {
        if (!creados || !inicializados) { registrar("Primero crea y llena los arreglos."); return; }
        const char* nombres[] = {"", "", "SUMA", "RESTA", "MULTIPLICACION", "CUADRADO"};
        int bloques = (tamano + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        registrar("Iniciando Kernel: " + string(nombres[opcion]));
        registrar("Bloques: " + to_string(bloques) + ", hilos/bloque: " + to_string(THREADS_PER_BLOCK));
        cudaEvent_t inicio, fin;
        CUDA_CHECK(cudaEventCreate(&inicio)); CUDA_CHECK(cudaEventCreate(&fin));
        CUDA_CHECK(cudaEventRecord(inicio));
        kernelAritmetica<<<bloques, THREADS_PER_BLOCK>>>(
            device_A, device_B, device_C, tamano, opcion, MODO_DETALLADO);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(fin)); CUDA_CHECK(cudaEventSynchronize(fin));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, inicio, fin));
        CUDA_CHECK(cudaEventDestroy(inicio)); CUDA_CHECK(cudaEventDestroy(fin));
        CUDA_CHECK(cudaMemcpy(host_C, device_C, (size_t)tamano * sizeof(long long), cudaMemcpyDeviceToHost));
        if (MODO_DETALLADO && tamano <= 40) {
            for (int i = 0; i < tamano; i++) {
                registrarSoloArchivo(prefijoDevice(i / THREADS_PER_BLOCK, i % THREADS_PER_BLOCK, i) +
                    "[Operacion: " + to_string(opcion) + "] Resultado: " + to_string(host_C[i]));
            }
        }
        registrar("Tiempo GPU (incluye printf si detalle=1): " + to_string(ms) + " ms");
    }

    void ejecutarReduccion(int opcion) {
        if (!creados || !inicializados) { registrar("Primero crea y llena los arreglos."); return; }
        // Reducimos A, como arreglo global seleccionado para las opciones 8-11.
        // Para B, se puede invocar el mismo kernel con device_B.
        int modo = (opcion == 10) ? 1 : ((opcion == 11) ? 2 : 0);
        string nombre = (opcion == 8) ? "SUMATORIA" : (opcion == 9) ? "PROMEDIO" :
                        (opcion == 10) ? "MAXIMO" : "MINIMO";
        int bloques = (tamano + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        long long* deviceParciales = nullptr;
        long long* hostParciales = new long long[bloques];
        CUDA_CHECK(cudaMalloc((void**)&deviceParciales, (size_t)bloques * sizeof(long long)));
        registrar("Reduccion " + nombre + " de A; N=" + to_string(tamano));
        registrar("Nivel 1 [DEVICE]: " + to_string(bloques) + " bloques, " +
                  to_string(THREADS_PER_BLOCK) + " hilos/bloque, __shared__");
        cudaEvent_t inicio, fin;
        CUDA_CHECK(cudaEventCreate(&inicio)); CUDA_CHECK(cudaEventCreate(&fin));
        CUDA_CHECK(cudaEventRecord(inicio));
        kernelReduccion<<<bloques, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(long long)>>>(
            device_A, deviceParciales, tamano, modo, MODO_DETALLADO);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(fin)); CUDA_CHECK(cudaEventSynchronize(fin));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, inicio, fin));
        CUDA_CHECK(cudaEventDestroy(inicio)); CUDA_CHECK(cudaEventDestroy(fin));
        CUDA_CHECK(cudaMemcpy(hostParciales, deviceParciales,
                              (size_t)bloques * sizeof(long long), cudaMemcpyDeviceToHost));

        // Replicamos en el archivo las entradas impresas por los hilos del kernel.
        // No modifica resultados: todas las cifras se obtienen de A y de los parciales GPU.
        if (MODO_DETALLADO && tamano <= 40) {
            for (int i = 0; i < tamano; i++) {
                registrarSoloArchivo(prefijoDevice(i / THREADS_PER_BLOCK, i % THREADS_PER_BLOCK, i) +
                    "[Operacion: Reduccion " + to_string(modo) + " - Carga] Valor: " + to_string(host_A[i]));
            }
            for (int b = 0; b < bloques; b++) {
                registrarSoloArchivo(prefijoDevice(b, 0, b * THREADS_PER_BLOCK) +
                    "[Operacion: Reduccion " + to_string(modo) + " - Parcial] Resultado: " + to_string(hostParciales[b]));
            }
        }

        // NIVEL 2 [HOST]: consolidar UN resultado por bloque.
        long long global = hostParciales[0];
        for (int b = 0; b < bloques; b++) {
            if (MODO_DETALLADO && bloques <= 40)
                registrar("Parcial [Bloque CUDA: " + to_string(b) + "] = " + to_string(hostParciales[b]));
            if (b == 0) continue;
            if (modo == 0) global += hostParciales[b];
            else if (modo == 1 && hostParciales[b] > global) global = hostParciales[b];
            else if (modo == 2 && hostParciales[b] < global) global = hostParciales[b];
        }
        if (opcion == 9) {
            double promedio = (double)global / tamano;
            registrar("Resultado global PROMEDIO(A): " + to_string(promedio));
        } else {
            registrar("Resultado global " + nombre + "(A): " + to_string(global));
        }
        registrar("Tiempo GPU del kernel de reduccion: " + to_string(ms) + " ms");
        CUDA_CHECK(cudaFree(deviceParciales)); delete[] hostParciales;
    }

    void liberarMemoria() {
        if (!creados) return;
        delete[] host_A; delete[] host_B; delete[] host_C;
        CUDA_CHECK(cudaFree(device_A)); CUDA_CHECK(cudaFree(device_B)); CUDA_CHECK(cudaFree(device_C));
        host_A = host_B = nullptr; host_C = nullptr;
        device_A = device_B = nullptr; device_C = nullptr;
        creados = false; inicializados = false;
        registrar("Memoria Host y Device liberada.");
    }
    ~OperacionesArreglos() {
        liberarMemoria();
        if (log.is_open()) log.close();
    }
};

int main() {

  cout << "[EQUIPO 7] INTEGRANTES:\n GOMEZ ENRIQUEZ MARIA FERNANDA\n LOZANO PEREZ JOSE MANUEL\n MARTINEZ ARIAS EDUARDO CALEB\n" << endl;

    static_assert(THREADS_PER_BLOCK > 0 && THREADS_PER_BLOCK <= 1024 &&
                  (THREADS_PER_BLOCK & (THREADS_PER_BLOCK - 1)) == 0,
                  "THREADS_PER_BLOCK debe ser potencia de 2 y no superar 1024");
    srand((unsigned int)time(nullptr));
    int cuenta = 0;
    CUDA_CHECK(cudaGetDeviceCount(&cuenta));
    if (cuenta == 0) { cerr << "[HOST] No hay GPU CUDA disponible\n"; return 1; }
    cudaDeviceProp gpu;
    CUDA_CHECK(cudaGetDeviceProperties(&gpu, 0));
    if (THREADS_PER_BLOCK > gpu.maxThreadsPerBlock) {
        cerr << "[HOST] THREADS_PER_BLOCK excede el limite de la GPU\n"; return 1;
    }
    OperacionesArreglos operaciones(N);
    operaciones.registrar("PRACTICA 1.3 - REDUCCIONES CUDA");
    operaciones.registrar(string("GPU: ") + gpu.name);
    operaciones.registrar("Memoria GPU: " + to_string(gpu.totalGlobalMem / 1048576.0) + " MB");
    operaciones.registrar("Maximo hilos por bloque: " + to_string(gpu.maxThreadsPerBlock));
    operaciones.registrar("Configuracion: N=" + to_string(N) + ", hilos=" +
                         to_string(THREADS_PER_BLOCK) + ", detalle=" + to_string(MODO_DETALLADO));
    int opcion;
    while (true) {
        operaciones.menu();
        if (!(cin >> opcion)) {
            if (cin.eof()) break;
            cin.clear(); cin.ignore(numeric_limits<streamsize>::max(), '\n');
            operaciones.registrar("Entrada invalida."); continue;
        }
        operaciones.registrar("Opcion seleccionada: " + to_string(opcion));
        if (opcion == 1) operaciones.crearArregloCUDA();
        else if (opcion >= 2 && opcion <= 5) operaciones.ejecutarOperacion(opcion);
        else if (opcion == 6) operaciones.llenarSecuencial();
        else if (opcion == 7) operaciones.llenarAleatorio();
        else if (opcion >= 8 && opcion <= 11) operaciones.ejecutarReduccion(opcion);
        else if (opcion == 12) { operaciones.registrar("Finalizando programa."); break; }
        else operaciones.registrar("Opcion no valida.");
    }
    
    cout << "[EQUIPO 7] INTEGRANTES:\n GOMEZ ENRIQUEZ MARIA FERNANDA\n LOZANO PEREZ JOSE MANUEL\n MARTINEZ ARIAS EDUARDO CALEB\n" << endl;
    return 0;
}
