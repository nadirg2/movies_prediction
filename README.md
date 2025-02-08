# Movies Prediction ML Pipeline

Проект рекомендательной системы на факторизационной машине, использующий Apache Airflow для управления пайплайном, включающим этапы предобработки данных, обучения модели и оценки модели.

## Структура проекта

```plaintext
project/
├── app/
│   ├── dags/
│   │   ├── ml_pipeline.py
│   ├── scripts/
│   │   ├── data_preprocessing.py
│   │   ├── model_training.py
│   │   ├── model_evaluation.py
│   ├── utils/
│   │   ├── logger.py
│   │   └── pickler.py
├── data/
│   ├── raw/
│   │   ├── rating.csv
│   │   ├── movie.csv
│   ├── processed/
│   │   ├── train_interactions.pkl
│   │   ├── test_interactions.pkl
│   │   ├── item_features.pkl
│   │   ├── index_to_movie_id.pkl
├── logs/
├── models/
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── data_analysis.ipynb
│   ├── model_training.ipynb
│   ├── model_evaluation.ipynb
├── venv/
├── requirements.txt
├── .gitignore
└── README.md
```
## Установка
1. Клонируйте репозиторий:
```
git clone https://github.com/yourusername/movies_prediction.git
cd movies_prediction
```
2. Установите зависимости:
```
pip install -r requirements.txt
```
3. Установите и настройте Apache Airflow:

```
pip install apache-airflow
airflow db init
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com
```

Настройте переменные окружения для Airflow:
```
export AIRFLOW_HOME=~/airflow
export AIRFLOW__CORE__DAGS_FOLDER=~/airflow/dags
```
Создайте директорию для DAGs и скопируйте ваш DAG файл:
```
export AIRFLOW_HOME=~/airflow
export AIRFLOW__CORE__DAGS_FOLDER=~/airflow/dags
```
## Запуск
1. Запустите веб-сервер Airflow:
```
airflow webserver --port 8080
```

2. В другом терминале запустите планировщик Airflow:
```
airflow scheduler
```

3. Откройте веб-интерфейс Airflow в браузере по адресу `http://localhost:8080` и войдите с учетными данными (например, `admin` / `admin`).

4. Найдите ваш DAG `ml_pipeline`, активируйте его и запустите.

## Скрипты
__data_preprocessing.py__

Этот скрипт выполняет предобработку данных, включая загрузку данных, преобразование временных меток, создание признаков жанров и генерацию матриц взаимодействий и признаков элементов.

__model_training.py__

Этот скрипт обучает модель LightFM на тренировочных данных и сохраняет обученную модель.

__model_evaluation.py__

Этот скрипт оценивает производительность модели с использованием метрик Precision@K и AUC на тестовых данных.

## Логи
Логи сохраняются в директорию logs, которая создается в текущей рабочей директории.

## Контакты
Если у вас есть вопросы или предложения, пожалуйста, свяжитесь с нами по адресу [gulievnadir3@gmail.com](mailto:gulievnadir3@gmail.com).
