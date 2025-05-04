## Related Repositories

- [DAGsHub Repo (Experiments)](https://dagshub.com/Maoelan/amazon-sentiment-analysis/experiments)
- [Automated Preprocessing Repo](https://github.com/Maoelan/Eksperimen_SML_Maulana-Muhammad)
- [CI/CD Workflow Repo](https://github.com/Maoelan/Workflow_CI_Maulana-Muhammad)
- [Docker Hub Repository](https://hub.docker.com/repository/docker/maoelana/amazon-sentiment-analysis/general)

## Additional Links

- [Google Drive Resources](https://drive.google.com/drive/u/0/folders/1mG6jjn8anwVlm3reWUgsSwCF-NuUyuPA)

## Deploy & Running Docker

**Build the Docker container from the registered MLflow model**
```bash
mlflow models build-docker -m "models:/Amazon-Sentiment-Analysis/1" --name "amazon-sentiment-analysis"
```

**Run the Docker container on port 8080**  
```bash
docker run -p 127.0.0.1:8080:8080 amazon-sentiment-analysis
```

**Pull the Docker image from Docker Hub and run it on port 5005:8080**
```bash
run docker : docker run --name amazon-sentiment-analysis-hub -p 5005:8080 maoelana/amazon-sentiment-analysis:latest
```
## Monitoring

**Prometheus port**
```bash
http://127.0.0.1:9090
```

**Grafana port**
```bash
http://127.0.0.1:3000
```

**Run flask (prometheus_exporter.py)**
```bash
python prometheus_exporter.py
```

**Run prometheus**
```bash
prometheus --config.file=prometheus.yml
```