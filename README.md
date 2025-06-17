📰 **BART-Base Summarization App**

A lightweight text summarization service using [BART-Base](https://huggingface.co/facebook/bart-base), fine‑tuned on the CNN/DailyMail dataset. Includes model training, a FastAPI web app, and GPU‑ready Docker deployment with load balancing.

---

## 🚀 Features

* 🔎 **Abstractive summarization** using BART‑Base
* 🖥️ **FastAPI** web interface with GPU acceleration
* 📦 **Dockerized** with Nginx load balancing for scalability
* 📊 Integrated **W\&B logging**
* ⚙️ Config‑driven training pipeline via `config.yaml`

---

## 🧠 Model

We use the `facebook/bart-base` model from Hugging Face, fine‑tuned on the CNN/DailyMail dataset for 3 epochs with early stopping and warm‑up steps.

## 🏗️ Project Structure

```
bart-summarization/
├── app/                  # FastAPI app
│   ├── main.py           # API logic
│   ├── requirements.txt  # API dependencies
│   ├── Dockerfile        # Docker for FastAPI + GPU
│   ├── nginx.conf        # Nginx load balancer
│   └── static/           # Frontend (optional)
│       └── index.html
├── training/             # Training code
│   ├── train.py
│   ├── config.yaml
│   └── utils.py
├── model/                # Saved BART weights & tokenizer
├── docker-compose.yml
└── README.md
```

---

## 🧪 Training

You can train the model using the script in `training/train.py`:

```bash
cd training
pip install -r requirements.txt
python train.py --config config.yaml
```

✅ Logs and metrics are tracked via [Weights & Biases](https://wandb.ai/). Early stopping is applied based on validation loss.

---

## 🌐 Serving the Model

### Start the app locally:

```bash
docker-compose up --build
```

The summarization API will be available at [http://localhost](http://localhost).

---

## 📩 API Usage

```bash
curl -X POST http://localhost/summarize \
     -H "Content-Type: application/json" \
     -d '{"text": "Your long article here..."}'
```

Returns:

```json
{ "summary": "Short summary of the input." }
```

---

## 🧊 Load Balancing

**Nginx** is used as a reverse proxy to forward requests to multiple Uvicorn workers, enabling parallel handling of incoming requests.

---

## 🐳 Docker Notes

* The app Dockerfile is based on `nvidia/cuda` and enables GPU inference.
* The `docker-compose.yml` mounts `./model` for the FastAPI container.

---

## 📈 ROUGE Scores

Using `facebook/bart-base` fine‑tuned on CNN/DailyMail (100000 size subsample): <br>
ROUGE‑1 (F₁): ≈ 41.23 <br>
ROUGE‑2 (F₁): ≈ 19.56 <br>
ROUGE‑L (F₁): ≈ 38.97 <br>
