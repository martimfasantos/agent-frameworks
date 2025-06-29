# Google ADK

- Repo: https://github.com/google/adk-python
- Documentation: https://google.github.io/adk-docs/

## Google ADK Examples

### How to setup

#### Virtual environment

Create a simple virtual environment with:

```bash
python3 -m venv .venv
```

Then activate it with:
```bash
# On Linux/macOS
source .venv/bin/activate
# On Windows
.venv\Scripts\activate
```

And install the requirements with:
```bash
pip install -r requirements.txt
```

#### .env

See .env.example and create a .env (on the root of the repository).
You need to get an open AI endpoint and key and fill them in.

#### Google API Key

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).

2. Create a new project or select an existing one.

3. Navigate to the **APIs & Services** section.

4. Click on **Credentials** in the left sidebar.

5. Click on **Create credentials** and select **API key**.

6. Copy the generated API key.

7. Enable the **Generative Language API** (which powers Google's AI models like Gemini) by navigating
to this [link](https://console.developers.google.com/apis/api/generativelanguage.googleapis.com/overview)

#### Run the examples

To run the examples, you should go to `google_adk` directory and run:

```bash
adk run <folder_name>
```