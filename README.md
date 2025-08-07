README
Predicción de Fuga de Clientes (Churn) en Telecom X
🎯 Objetivo del Proyecto
Este proyecto de Machine Learning tiene como objetivo principal desarrollar un pipeline robusto para predecir la fuga de clientes (churn) en Telecom X. Al identificar proactivamente a los clientes con alta probabilidad de cancelar sus servicios, la empresa puede implementar estrategias de retención dirigidas, salvaguardar sus ingresos y fomentar un crecimiento sostenido.
📊 Conjunto de Datos
El análisis se realizó utilizando el conjunto de datos TelecomX_Clean.csv. Este dataset contiene registros de clientes con diversas características que describen sus servicios de telecomunicaciones y comportamiento.
Columnas Clave:
•	Churn: Variable objetivo (Sí/No) que indica si el cliente ha cancelado su servicio.
•	customer_gender, customer_SeniorCitizen, customer_Partner, customer_Dependents: Atributos demográficos.
•	customer_tenure: Antigüedad del cliente en la empresa.
•	phone_PhoneService, phone_MultipleLines: Detalles del servicio telefónico.
•	internet_InternetService, internet_OnlineSecurity, internet_OnlineBackup, internet_DeviceProtection, internet_TechSupport, internet_StreamingTV, internet_StreamingMovies: Detalles de los servicios de internet.
•	account_Contract, account_PaperlessBilling, account_PaymentMethod: Detalles del contrato y facturación.
•	account_Charges_Monthly, account_Charges_Total: Cargos mensuales y totales.
📁 TelecomX-Churn-Prediction
│
├── data/
│ └── TelecomX_Clean.csv # Dataset original (anonimizado)
│
├── notebooks/
│ └── Exploratory_Analysis.ipynb # Análisis exploratorio + visualizaciones
│
├── pipeline/
│ └── churn_pipeline.py # Preprocesamiento + entrenamiento + predicción
│
├── app/
│ └── app.py # API REST con FastAPI para predicción
│ └── Dockerfile # Contenedor para desplegar la app
│
├── docs/
│ └── Informe_Modelado.pdf # Informe técnico con análisis y recomendaciones
│
└── README.md # Documentación general del proyecto

🚀 Pipeline de Machine Learning
El pipeline de Machine Learning se construyó siguiendo las mejores prácticas para el modelado predictivo:
1.	Preparación y Limpieza de Datos:
o	Manejo de Valores Ausentes: Se trataron los valores ausentes en account_Charges_Total (espacios en blanco convertidos a NaN y luego imputados con la mediana) para asegurar la integridad numérica.   
o	Codificación de Variables Categóricas: Las características categóricas se transformaron a un formato numérico utilizando One-Hot Encoding, lo que permite a los modelos procesarlas sin introducir relaciones ordinales artificiales. Se prestó especial atención a categorías como "No phone service" y "No internet service", tratándolas como segmentos distintos.   
o	Escalado de Características Numéricas: Las variables numéricas (customer_tenure, account_Charges_Monthly, account_Charges_Total) se escalaron utilizando RobustScaler para normalizar sus rangos y mejorar la convergencia y el rendimiento de los algoritmos, siendo robusto a valores atípicos.   
o	Manejo del Desequilibrio de Clases: Dado que la fuga de clientes es un evento minoritario, se aplicó SMOTE (Synthetic Minority Over-sampling Technique) para equilibrar la distribución de la clase objetivo (Churn), lo que ayuda a los modelos a aprender patrones efectivos de ambas clases.   
2.	Análisis de Correlación y Selección de Variables:
o	Se realizó un análisis de correlación para comprender las relaciones entre las características y la variable objetivo.
o	La selección de características se basó en la interpretabilidad de los coeficientes de la Regresión Logística y la importancia de las características del Bosque Aleatorio, buscando un equilibrio entre poder predictivo e interpretabilidad.   
3.	Entrenamiento de Modelos de Clasificación:
o	El conjunto de datos se dividió en un 80% para entrenamiento y un 20% para prueba, utilizando muestreo estratificado para mantener la proporción de clases en ambos conjuntos.   
o	Se entrenaron dos modelos predictivos:
	Regresión Logística: Elegida por su interpretabilidad, que permite entender la dirección y magnitud del impacto de cada factor en la probabilidad de fuga.   
	Bosque Aleatorio: Un método de conjunto robusto, capaz de capturar relaciones no lineales complejas y proporcionar puntuaciones de importancia de características.   
4.	Evaluación del Rendimiento del Modelo:
o	Los modelos se evaluaron utilizando métricas clave para problemas de clasificación desequilibrados, incluyendo:
	Precisión (Precision), Sensibilidad (Recall) y Puntuación F1 (F1-Score): Cruciales para evaluar el rendimiento en la clase minoritaria (fuga).   
	Curva ROC y AUC (Área bajo la Curva): Para medir la capacidad general del modelo para distinguir entre clientes que abandonan y los que no.   
	Matriz de Confusión: Para visualizar los tipos específicos de errores (Verdaderos Positivos, Falsos Negativos, etc.).   
💡 Conclusión Estratégica y Factores Clave de Fuga
El análisis exhaustivo de los modelos de Regresión Logística y Bosque Aleatorio reveló los siguientes factores como los más influyentes en la fuga de clientes de Telecom X:
•	Tipo de Contrato (Mes a Mes): Los clientes con contratos mensuales muestran una probabilidad significativamente mayor de fuga, lo que sugiere una menor lealtad y facilidad para cambiar de proveedor.
•	Antigüedad del Cliente (customer_tenure): Una menor antigüedad en la empresa se correlaciona fuertemente con una mayor propensión a la fuga, destacando la importancia de las estrategias de incorporación y retención temprana.
•	Servicio de Internet (Fibra Óptica): Los clientes con servicio de fibra óptica parecen tener una mayor propensión a la fuga, lo que podría indicar problemas de calidad percibida o una mayor sensibilidad a las ofertas de la competencia en este segmento.
•	Método de Pago (Cheque Electrónico): El uso del cheque electrónico como método de pago se asocia con una mayor probabilidad de fuga, lo que podría ser un indicador de un segmento de clientes con menor fidelización.
•	Ausencia de Servicios Adicionales (Seguridad en Línea, Soporte Técnico): La falta de servicios "adhesivos" como seguridad en línea o soporte técnico contribuye a la fuga, ya que estos aumentan el valor percibido y el compromiso del cliente.
•	Cargos Mensuales (account_Charges_Monthly): Cargos mensuales más altos pueden indicar un mayor riesgo de fuga, especialmente si los clientes no perciben un valor proporcional a lo que pagan.
Estos hallazgos proporcionan información valiosa para que Telecom X desarrolle estrategias de retención personalizadas y proactivas, enfocándose en los segmentos de clientes más vulnerables y abordando los factores subyacentes de la fuga.
🛠️ Tecnologías y Librerías Utilizadas
•	Python
•	Pandas: Manipulación y análisis de datos.
•	NumPy: Operaciones numéricas.
•	Scikit-learn: Preprocesamiento de datos, modelos de clasificación y métricas de evaluación.
•	Imbalanced-learn (imblearn): Manejo del desequilibrio de clases (SMOTE).
•	Matplotlib y Seaborn: Visualización de datos y resultados del modelo.

🏃‍♀️ Cómo Ejecutar el Código
1.	Clonar el Repositorio: Descargue o clone este repositorio en su máquina local.
2.	Preparar el Entorno: Se recomienda usar un entorno virtual. Instale las librerías necesarias:bash pip install pandas numpy scikit-learn imblearn matplotlib seaborn
3.	Cargar el Dataset: Asegúrese de que el archivo TelecomX_Clean.csv esté en la misma carpeta que el script o notebook.
4.	Ejecutar el Notebook/Script: Puede ejecutar el código paso a paso en un entorno de Jupyter Notebook (como Google Colaboratory) o como un script de Python. Si usa Google Colab, recuerde subir el archivo TelecomX_Clean.csv a su sesión.
   ✍️ Autor Dr. Martin Abreu
    Especialista en planificación educativa e inteligencia artificial.
   📫 Contacto: LinkedIn | GitHub


