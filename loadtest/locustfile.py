from locust import HttpUser, task, between

#uses locust to load test
PAYLOAD = {"x": [0.1] * 1024}

class ApiUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(5)
    def health(self):
        self.client.get("/health")

    @task(1)
    def predict(self):
        with self.client.post("/predict", json=PAYLOAD, catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"status={r.status_code}, body={r.text}")
