from flask import Flask, request, jsonify, Response
import requests
import time
import psutil
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client.exposition import basic_auth_handler
from prometheus_client import multiprocess

app = Flask(__name__)

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests')
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP Request Latency')
REQUEST_FAILURES = Counter('http_requests_failures_total', 'Total failed HTTP requests')
REQUEST_SUCCESSES = Counter('http_requests_success_total', 'Total successful HTTP requests')
LAST_STATUS_CODE = Gauge('http_response_status_code', 'Last HTTP response status code')

CPU_USAGE = Gauge('system_cpu_usage', 'CPU Usage %')
RAM_USAGE = Gauge('system_ram_usage', 'RAM Usage %')
NET_SENT = Gauge('network_io_sent_bytes', 'Network bytes sent')
NET_RECV = Gauge('network_io_recv_bytes', 'Network bytes received')
PROCESS_COUNT = Gauge('system_process_count', 'Number of running processes')

@app.route('/metrics', methods=['GET'])
def metrics():
    CPU_USAGE.set(psutil.cpu_percent(interval=1))
    RAM_USAGE.set(psutil.virtual_memory().percent)
    NET_SENT.set(psutil.net_io_counters().bytes_sent)
    NET_RECV.set(psutil.net_io_counters().bytes_recv)
    PROCESS_COUNT.set(len(psutil.pids()))

    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    print("Incoming request...")
    REQUEST_COUNT.inc()

    api_url = "http://127.0.0.1:5005/invocations"
    data = request.get_json()

    try:
        response = requests.post(api_url, json=data)
        duration = time.time() - start_time
        REQUEST_LATENCY.observe(duration)

        status_code = response.status_code
        LAST_STATUS_CODE.set(status_code)

        if status_code == 200:
            REQUEST_SUCCESSES.inc()
        else:
            REQUEST_FAILURES.inc()

        return jsonify(response.json())

    except Exception as e:
        REQUEST_FAILURES.inc()
        LAST_STATUS_CODE.set(500)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)