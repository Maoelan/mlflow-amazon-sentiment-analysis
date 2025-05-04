https://dagshub.com/Maoelan/amazon-sentiment-analysis/experiments < Dagshub repo

https://github.com/Maoelan/Eksperimen_SML_Maulana-Muhammad < Automate Preprocessing repo

https://github.com/Maoelan/Workflow_CI_Maulana-Muhammad < Worflow CI/CD repo

mlflow models build-docker -m "models:/Amazon-Sentiment-Analysis/1" --name "amazon-sentiment-analysis" < Build & Run Docker Container

docker run -p 127.0.0.1:8080:8080 amazon-sentiment-analysis < Use only port 8080 because windows will block another port