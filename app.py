from flask import Flask, render_template, request
from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect

app = Flask(__name__)

# cache models so they don't reload every request
loaded_models = {}

def load_model(src, tgt):
    model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"

    if model_name in loaded_models:
        return loaded_models[model_name]

    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        loaded_models[model_name] = (tokenizer, model)
        return tokenizer, model
    except:
        return None, None

def translate_text(text, target_lang, source_lang=None):
    
    try:
        # detect language only in auto mode
        if source_lang is None:
            source_lang = detect(text)

        # langdetect mistake fix for short words
        if len(text.strip()) <= 5:
            source_lang = "en"

        # if same language
        if source_lang == target_lang:
            return text

        tokenizer, model = load_model(source_lang, target_lang)

        # fallback: try english if pair not available
        if model is None:
            tokenizer, model = load_model("en", target_lang)
            source_lang = "en"

        if model is None:
            return f"Translation not supported: {source_lang} → {target_lang}"

        inputs = tokenizer(text, return_tensors="pt", padding=True)
        translated = model.generate(**inputs)
        result = tokenizer.decode(translated[0], skip_special_tokens=True)

        return result

    except Exception as e:
        return "Error detecting language or translating." 

@app.route('/', methods=['GET','POST'])
def index():
    translated_text = ""

    if request.method == 'POST':
        text = request.form['data']
        target_lang = request.form['target_lang']

        # check auto detect
        auto_detect = request.form.get('auto_detect')

        if auto_detect == "yes":
            source_lang = None
        else:
            source_lang = request.form['source_lang']

        translated_text = translate_text(text, target_lang, source_lang)

    return render_template('index.html', translated_text=translated_text)


if __name__ == '__main__':
    app.run(debug=True)