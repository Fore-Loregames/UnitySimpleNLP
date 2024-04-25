using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public static class NLPUtilities
{


    public static string[] Tokenize(string sentence)
    {
        char[] separators = new char[] { ' ', '.', ',', '!', '?', ':' };
        return sentence.ToLower().Split(separators, StringSplitOptions.RemoveEmptyEntries);
    }

    public static int[] BagOfWords(string sentence, List<string> vocabulary)
    {
        string[] tokens = Tokenize(sentence);
        int[] bag = new int[vocabulary.Count];
        foreach (var word in tokens)
        {
            for (int i = 0; i < vocabulary.Count; i++)
            {
                if (word == vocabulary[i])
                {
                    bag[i] = 1;
                    break;
                }
            }
        }
        return bag;
    }
}
