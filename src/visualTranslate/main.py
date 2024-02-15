from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


if __name__ == "__main__":
    
    # Load the model and tokenizer

    checkpoint = "facebook/nllb-200-1.3B"

    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    source_lang = "jpn_Jpan"
    target_lang = "eng_Latn"

    translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang=source_lang, tgt_lang=target_lang, max_length = 400)


    text = "おはようございます。天気が良くて空がきれいです。"
    text2 = "古池や。蛙飛び込む。水の音" # The old pond. A frog leaps in. Sound of the water.
    text3 = "白露に 風の吹きしく 秋の野は" # In the autumn fields the wind blows briskly on the white dew.

    output = translator(text) # Good morning. The weather is fine and the sky is clear.
    output2 = translator(text2) # The frog leaps into the old pond. The sound of water.
    output3 = translator(text3) # In the autumn fields, the wind blows briskly on the white dew.
    translated_text = output[0]['translation_text']
    translated_text2 = output2[0]['translation_text']   
    translated_text3 = output3[0]['translation_text']
    print(f"Frase 1: {translated_text}")
    print(f"Frase 2: {translated_text2}")
    print(f"Frase 3: {translated_text3}")
