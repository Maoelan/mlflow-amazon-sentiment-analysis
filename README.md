## 🔗 Related Repositories

- [DAGsHub Repo (Experiments)](https://dagshub.com/Maoelan/amazon-sentiment-analysis/experiments)
- [Automated Preprocessing Repo](https://github.com/Maoelan/Eksperimen_SML_Maulana-Muhammad)
- [CI/CD Workflow Repo](https://github.com/Maoelan/Workflow_CI_Maulana-Muhammad)

## 🌐 Additional Links

- [Docker Hub Repository](https://hub.docker.com/repository/docker/maoelana/amazon-sentiment-analysis/general)
- [Google Drive Resources](https://drive.google.com/drive/u/0/folders/1mG6jjn8anwVlm3reWUgsSwCF-NuUyuPA)

## 🐳 Deploy Using Docker

### Build the Docker container from the registered MLflow model
```bash
mlflow models build-docker -m "models:/Amazon-Sentiment-Analysis/1" --name "amazon-sentiment-analysis"
```

### Run the Docker container on port 8080  
(Use only port 8080 since Windows might block other ports)
```bash
docker run -p 127.0.0.1:8080:8080 amazon-sentiment-analysis
```