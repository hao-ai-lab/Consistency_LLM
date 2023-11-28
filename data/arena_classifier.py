from transformers import pipeline
import json
import random

classify_by = None # "language" or "topic"
# path of arena dataset
filename = "/data/clean_arena_new.json"
with open(filename, "r") as f:
    data = json.load(f)

#################################### Classify by Languages #############################
if classify_by == "language":
    languages = ["English", "Spanish", "Russian", "Japanese", "Portuguese"]
    for language in languages:
        data = [d for d in data if d['language'] == language]
        with open(f"raw_data/{language}.json", "w") as f:
            json.dump(data, f)

#################################### Classify by Topics ####################################
if classify_by == "topic":
    classes = {}
    count = 0
    classifier = pipeline("text-classification", model="alimazhar-110/website_classification")
    for i, d in enumerate(data):
        try:
            if d['language'] != 'English':
                continue
            prompt = d['conversation'][0]['content']
            out = classifier(prompt)
            label = out[0]['label']
            if label not in classes:
                classes[label] = []
            classes[label].append(d)
        except:
            count += 1
            print(f"ignore {count}/{i}")

    for c in classes:
        print(c, len(classes[c]))
        filename = c.replace("/", "_")
        with open(f"raw_data/{filename}.json", "w") as f:
            json.dump(classes[c], f)

#################################### Mix Topics ####################################
mix_topics = ["Business_Corporate", "Computers and Technology", 
              "Games", "Education", "Social Networking and Messaging"]
all_topic_data = []
for topic in mix_topics:
    with open(f"raw_data/{topic}.json", "r") as f:
        data = json.load(f)[:5000]
        for d in data:
            d['topic'] = topic
        print(topic, len(data))
        all_topic_data += data
print(len(all_topic_data))
random.shuffle(all_topic_data)
with open(f"raw_data/mix_topics.json", "w") as f:
    json.dump(all_topic_data, f)