# İşçi Sağlığı ve Güvenliği: Hareketsizlil Detectörü

This is a Python repository for the Worker Health and Safety (İşçi Sağlığı ve Güvenliği) project that focuses on detecting NO-MOTINON

## Kurulum  - Installation

1. Python kurulu olduğundan emin olun, - First, ensure that you have Python installed on your system.

```
python --version
```

2. Github tan klonlayın - Clone the repository:

```
git clone https://github.com/cappittall/motion.git
```


3. Sanal ortam oluşturma - Create a Python virtual environment:

```
cd motion
python3 -m venv myenv
```


4. Sanal ortamı aktive etme - Activate the virtual environment:

- Windows:

  ```
  myenv\Scripts\activate
  ```

- macOS and Linux:

  ```
  source myenv/bin/activate
  ```

5. Gerekli paketlerin kurulumu - Install the required dependencies:

```
pip install -r requirements.txt
```



## Aplikasyonu çalıştırma - Running the Application

1. FastAPI yi çalıştırma - Start the FastAPI server:

```
pip install "uvicorn[standard]"

uvicorn detect:app --reload
```

2. Web tarayıcıyı açın ve tarayıcıda adresine gidin [localhost:8000](http://localhost:8000) -  Open your web browser and navigate to [localhost:8000](http://localhost:8000) 

klasörü altında kaydedilir

## Ek açıklamalar - Instructions

Ayrıca [40 pinli](https://coral.ai/docs/dev-board/datasheet/) giriş çıkış pinlerinden alarm yada başka bir sistemi tetiklemek için gerekli sinyal çıkışı alınabilecektir. Bu son ayarlamalar cihaz üzerinde yapılacaktır. 
