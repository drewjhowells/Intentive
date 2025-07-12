🗣️ Tone Transformation Dataset

Category: NLP / Emotion Classification / Style Transfer
Author: Gopi Krishnan R
License: CC BY-SA 4.0
📘 Description

This dataset contains English sentences labeled with emotional tone and transformed into three distinct communication styles:

    Polite

    Professional

    Casual

Each row includes:

    An original sentence

    Its associated emotion (e.g., joy, sadness, anger, love)

    Tone-transformed versions in polite, professional, and casual forms

🧠 Use Cases

This dataset is suitable for:

    Emotion detection

    Text tone/style transfer models

    Chatbot tone adaptation

    Multi-tone response generation

    Prompt engineering tasks

📁 Data Format
Original	Polite	Professional	Casual	Emotion
i feel incredibly lucky just to be able to talk to her;joy	I feel incredibly fortunate...	I am extremely grateful...	I feel so lucky...	joy

    Note: If the emotion is embedded in the Original column, you can extract it using Python:
    text, emotion = text.rsplit(";", 1)

📚 Source & Attribution

This dataset is a derivative of the Emotions Dataset for NLP by Praveen Govi.

From the original dataset, only the English sentences and emotion labels were used.
All tone-transformed variants (Polite, Professional, Casual) were authored by Gopi Krishnan R.
📄 License

Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)

You are free to:

    Share — copy and redistribute the dataset

    Adapt — remix, transform, and build upon it, even commercially

Under the following terms:

    Attribution — You must credit the original and derivative author

    ShareAlike — You must license derivatives under the same terms