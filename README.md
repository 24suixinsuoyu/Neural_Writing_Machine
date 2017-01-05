# NeuralWritingMachine
![image](https://github.com/zzw922cn/NeuralWritingMachine/blob/master/main_page.png)
**Neural Writing Machine (NWM)** can write styles of documents such like *Chinese poems*, *couplets*, *novel*, *lyrics of Jay Chou or Xi Lin*. NWM is based on end-to-end RNN encoder-decoder approach, and can be combined with attention mechanism. For more details, please visit my wechat public media:**deeplearningdigest**.

We provide four types of text databases, which are:
+ Chinese Poems
+ Chinese Couplets
+ Chinese Novels
+ Lyrics of Jay Chou and Xi Lin

Usage of NWM is as follows:

```
usage: train.py [-h] [--style STYLE] [--data_dir DATA_DIR]
                [--save_dir SAVE_DIR] [--log_dir LOG_DIR]
                [--rnn_size RNN_SIZE] [--embedding_size EMBEDDING_SIZE]
                [--num_layers NUM_LAYERS] [--model MODEL] [--rnncell RNNCELL]
                [--attention ATTENTION] [--batch_size BATCH_SIZE]
                [--seq_length SEQ_LENGTH] [--num_epochs NUM_EPOCHS]
                [--save_every SAVE_EVERY] [--grad_clip GRAD_CLIP]
                [--learning_rate LEARNING_RATE] [--decay_rate DECAY_RATE]
                [--keep KEEP] [--pretrained PRETRAINED]

optional arguments:
  -h, --help            show this help message and exit
  --style STYLE         set the type of generating sequence,egs: poem,
                        couplet, novel, LX, FWS
  --data_dir DATA_DIR   set the data directory which contains new.txt
  --save_dir SAVE_DIR   set directory to store checkpointed models
  --log_dir LOG_DIR     set directory to store checkpointed models
  --rnn_size RNN_SIZE   set size of RNN hidden state
  --embedding_size EMBEDDING_SIZE
                        set size of word embedding
  --num_layers NUM_LAYERS
                        set number of layers in the RNN
  --model MODEL         set the model
  --rnncell RNNCELL     set the cell of rnn, eg. rnn, gru, or lstm
  --attention ATTENTION
                        set attention mode or not
  --batch_size BATCH_SIZE
                        set minibatch size
  --seq_length SEQ_LENGTH
                        set RNN sequence length
  --num_epochs NUM_EPOCHS
                        set number of epochs
  --save_every SAVE_EVERY
                        set save frequency while training
  --grad_clip GRAD_CLIP
                        set clip gradients when back propagation
  --learning_rate LEARNING_RATE
                        set learning rate
  --decay_rate DECAY_RATE
                        set decay rate for rmsprop
  --keep KEEP           init from trained model
  --pretrained PRETRAINED
                        init from pre-trained model
```

## Environment
- Tensorflow rc0.12
- Python2.7
- Numpy

## Training
If you train for the first time, you can input 
`python train.py --style=poem`

If you want to train from the restored model, you can input
`python train.py --style=poem --keep=True`

NWM would check for the model file and save to the specified folder automatically!

## Generation
You can sample from the trained model to generate new documents, usage is as follows:
```
usage: sample.py [-h] [--style STYLE] [--save_dir SAVE_DIR] [--n N]
                 [--start START] [--sample SAMPLE]

optional arguments:
  -h, --help           show this help message and exit
  --style STYLE        set the type of generating sequence,egs: novel, jay,
                       linxi, tangshi, duilian
  --save_dir SAVE_DIR  model directory to store checkpointed models
  --n N                number of words to sample
  --start START        prime text
  --sample SAMPLE      three choices:argmax,weighted,combined
```

There are three methods for sampling, they are `argmax`, `weighted` and `combined`.

`argmax `: generate the word with highest probability at each timestep

`weighted `: sample the word with weighted probability at each timestep

`combined `: generate the word with highest probability at each timestep except that when the previous word is a marker of full stop, which is `。`.

I suggest that you should choose the `weighted` sampling.
## Generation Examples
### Poem:
```
一春落去细复来，画眉峰上吴支路。一去紫露微月，夜深锁五月黑。
鸣筝鸟绿青子，风雨更吹魂。恨水来似花云尽，愁尽落灯红。
```
### FWS-style Lyric:

```
从前进开封 出水芙蓉加了星
在狂风叫我的爱情 让你更暖
心上人在脑海别让人的感觉
想哭 是加了糖果
船舱小屋上的锈却现有没有在这感觉
你的美 没有在
我却很专心分化
还记得
原来我怕你不在 我不是别人
要怎么证明我没力气 我不用 三个马蹄
生命潦草我在蓝脸喜欢笑
中古世界的青春的我
而她被芒草割伤
我等月光落在黑蝴蝶
外婆她却很想多笑对你不懂 我走也绝不会比较多
反方向左右都逢源不恐 只能永远读着对白
```


