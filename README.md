# Bee_app

Beside honeybee, many solitary bee species numbers are dwindling in the wild. It is hoped this app can assist in identifying bee species for tracking.

Using fastai, trained a CNN ResNet-18 model (FlattenedLoss of CrossEntropyLoss) on 295903 images of 99 bee species I sourced from https://species.biodiversityireland.ie/ list of bee species native to Ireland (and other geographical locations).

Streamlit bee classification app https://c123ian-bee-app-app-evtngf.streamlitapp.com/

Guide to creating streamlit app: https://ibronko.hashnode.dev/fastai-practical-deep-learning-for-coders-3
Guide to creating fastai model: https://towardsdatascience.com/how-to-create-an-app-to-classify-dogs-using-fastai-and-streamlit-af3e75f0ee28

Note: the bee dataset I created is highly imbalanced.
