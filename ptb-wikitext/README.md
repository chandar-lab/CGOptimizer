# Download and set up the training directory

A separate download is necessary for the PTB/Wikitext data.

After cloning the repository, create a Dataset directory in your `ptb-wikitext` directory. Code will search this directory by default.

```
cd CriticalGradientOptimization

cd ptb-wikitext

mkdir Dataset

cd Dataset
```

For PTB:

```
wget https://www.dropbox.com/s/rbuoo026sb276dk/ptb.zip

unzip ptb.zip
```

For WikiText:

```
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip

unzip ptb.zip
```