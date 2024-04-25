using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine;
namespace EveAI.Assets.Eve.scripts
{
    public class TestExampleUnity : MonoBehaviour
    {
        SimpleNeuralNetwork network;
        SimpleFeatureExtractor featureExtractor;
        List<string> vocabulary;
        List<string> tags;
        Dictionary<string, int> tagToIndex;
        public float learningRate = 0.1f;
        public int epoochs = 1000;
        public int hiddenSize = 100;
        RootObject rootObject;

        void Start()
        {

            Training();
            //example usage
            string incomingmessage = "Hello, how are you?";
            string tag = PredictTag(incomingmessage);
            Debug.Log(tag);

            foreach (var intent in rootObject.intents)
            {
                if (intent.tag == tag)
                {
                    string response = intent.responses[UnityEngine.Random.Range(0, intent.responses.Count)];

                    Debug.Log(response);

                    break;
                }
            }

        }

        void Training()
        {
            TextAsset textAsset = Resources.Load<TextAsset>("intents");
            string jsonString = textAsset.text;

            rootObject = JsonUtility.FromJson<RootObject>(jsonString);
            // Initialize Feature Extractor with training sentences
            List<string> trainingSentences = rootObject.intents.SelectMany(intent => intent.patterns).ToList();
            featureExtractor = new SimpleFeatureExtractor(trainingSentences, 2); 

            // Initialize the neural network
            int inputSize = featureExtractor.vocabulary.Count; 
            int outputSize = rootObject.intents.Count;
            network = new SimpleNeuralNetwork(inputSize, hiddenSize, outputSize);

            // Build tags list and map
            BuildTagsListAndMap(rootObject);

            // Prepare training data using the feature extractor
            var (trainingInputs, trainingOutputs) = PrepareTrainingData(rootObject);


            // Train the network
            network.Train(trainingInputs, trainingOutputs, epoochs, learningRate);
        }
        void BuildTagsListAndMap(RootObject _rootObject)
        {
            tags = _rootObject.intents.Select(intent => intent.tag).ToList();
            tagToIndex = tags.Select((tag, index) => new { tag, index })
                             .ToDictionary(t => t.tag, t => t.index);
        }

        void BuildVocabularyAndTags(RootObject rootObject)
        {
            HashSet<string> vocabSet = new HashSet<string>();
            HashSet<string> tagSet = new HashSet<string>();

            foreach (var intent in rootObject.intents)
            {
                foreach (var pattern in intent.patterns)
                {
                    foreach (var word in NLPUtilities.Tokenize(pattern))
                    {
                        vocabSet.Add(word);
                    }
                }
                tagSet.Add(intent.tag);
            }

            vocabulary = vocabSet.ToList();
            tags = tagSet.ToList();
            tagToIndex = tags.Select((tag, index) => new { tag, index })
                             .ToDictionary(t => t.tag, t => t.index);
        }
        (List<int[]>, List<int[]>) PrepareTrainingData(RootObject rootObject)
        {
            List<int[]> trainingInputs = new List<int[]>();
            List<int[]> trainingOutputs = new List<int[]>();

            foreach (var intent in rootObject.intents)
            {
                int[] output = new int[tags.Count];
                output[tagToIndex[intent.tag]] = 1;

                foreach (var pattern in intent.patterns)
                {
                    int[] input = featureExtractor.GetFeatures(pattern); 
                    trainingInputs.Add(input);
                    trainingOutputs.Add(output);
                }
            }

            return (trainingInputs, trainingOutputs);
        }


        public string PredictTag(string sentence)
        {
            Debug.Log("Predicting tag for sentence: " + sentence);
            int[] features = featureExtractor.GetFeatures(sentence); 
            float[] output = network.Forward(features);
            int maxIndex = output.ToList().IndexOf(output.Max());
            return tags[maxIndex];
        }
    }
}


[System.Serializable]
public class Intent
{
    public string tag;
    public List<string> patterns;
    public List<string> responses;
}

[System.Serializable]
public class RootObject
{
    public List<Intent> intents;
}
