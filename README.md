# NeuralWritingMachine
Neural Writing Machine (NWM) can write styles of documents such like *Chinese poems*, *couplets*, *novel*, *lyrics of Jay Chou or Xi Lin*. NWM is based on end-to-end RNN encoder-decoder approach, and can be combined with attention mechanism. For more details, please visit my wechat public media:**deeplearningdigest**.

We provide four types of text databases, which are:
+ Chinese Poems
+ Chinese Couplets
+ Chinese Novels
+ Lyrics of Jay Chou and Xi Lin

```python
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

预训练好的模型（batch_size=16, seq_length=16）下载地址：https://pan.baidu.com/s/1i5E2Vq9

## 环境
基于Tensorflow rc0.12版本，Python2.7，支持CPU/GPU。

## 训练
`python train.py `

如果要使用基于注意力的解码器，那么请在train.py文件中指定attention为True，或者在命令行使用如下：

`python train.py attention=True `

## 抽样生成单词
`python sample.py `

抽样有三种方法，分别是argmax、weighted、combined，它们的作用分别是

`argmax `:每一步都是根据分布概率得到概率最大的那个汉字

`weighted `:每一步都是根据概率分布抽样得到汉字

`combined `:每次遇到句号时，就会根据概率分布抽样；而其余情况都是得到概率最大的汉字

实际情况表明，使用`weighted `可以得到最佳的生成效果，而使用`argmax `或者 `combined `容易导致重复抽样的问题。

## 生成序列案例
使用`从前`作为种子序列，生成歌词如下：

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


