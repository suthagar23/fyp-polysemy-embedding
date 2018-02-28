# Polysemy Embedding
## Pre processing Model

We focused on these methods for the pre-processing
1. sent_tokenize
2. word_tokenzie
3. stop words removal
4. Add Synonyms List
5. Add Antonyms List
6. Lemmatization (Verb, Noun, Adjective, Adverb)

## Outut Structure
[
{
"2":{
"actualLine":"Hello this is a tutorial, on how to convert the word in an integer format. this is a beautiful daym Jack is going to office.\n",
"sentTokenize":[
"Hello this is a tutorial, on how to convert the word in an integer format.",
"this is a beautiful daym Jack is going to office."
],
"words":{
"wordTokenize":[],
"info":{
"0":{
"word":"this",
"isStopWord":true
},
"1":{},
"2":{},
"3":{
"word":"beautiful",
"isStopWord":false,
"synonymsCount":1,
"synonymsList":[
"beautiful"
],
"antonymsCount":1,
"antonymsList":[
"ugly"
],
"lemmatize":{
"verb":"beautiful",
"noun":"beautiful",
"adjective":"beautiful",
"adverb":"beautiful"
}
},
"4":{},
"5":{},
"6":{},
"7":{},
"8":{
"word":"to",
"isStopWord":true
},
"9":{},
"10":{}
}
}
}
}
]

