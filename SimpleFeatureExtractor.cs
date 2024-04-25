using System.Collections;
using System.Linq;
using System.Collections.Generic;
using UnityEngine;
using System;
public class SimpleFeatureExtractor
{
    public List<string> vocabulary;
    public int nGramSize;

    public SimpleFeatureExtractor(List<string> trainingSentences, int nGramSize = 2)
    {
        this.nGramSize = nGramSize;
        BuildVocabulary(trainingSentences);
    }

    private void BuildVocabulary(List<string> sentences)
    {
        HashSet<string> vocabSet = new HashSet<string>();
        foreach (var sentence in sentences)
        {
            var words = Tokenize(sentence);
            foreach (var word in words)
            {
                vocabSet.Add(word); // Add individual words

                // Add n-grams to vocabulary
                for (int n = 2; n <= nGramSize; n++)
                {
                    for (int i = 0; i < words.Length - n + 1; i++)
                    {
                        string nGram = string.Join(" ", words.Skip(i).Take(n));
                        vocabSet.Add(nGram);
                    }
                }
            }
        }
        vocabulary = vocabSet.ToList();
    }

    public int[] GetFeatures(string sentence)
    {
        var features = new int[vocabulary.Count];
        var words = Tokenize(sentence);

        foreach (var word in words)
        {
            int index = vocabulary.IndexOf(word);
            if (index != -1) features[index] = 1;

            for (int n = 2; n <= nGramSize; n++)
            {
                for (int i = 0; i < words.Length - n + 1; i++)
                {
                    string nGram = string.Join(" ", words.Skip(i).Take(n));
                    index = vocabulary.IndexOf(nGram);
                    if (index != -1) features[index] = 1;
                }
            }
        }

        return features;
    }

    private string[] Tokenize(string sentence)
    {
        char[] separators = new char[] { ' ', '.', ',', '!', '?', ':' };
        return sentence.ToLower().Split(separators, StringSplitOptions.RemoveEmptyEntries);
    }
}
