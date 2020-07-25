# ImageRecognition


# FastAI
![alt text](https://github.com/sagasu/ImageRecognition/blob/master/DogsAndCats.png?raw=true)

please run `run_vision_parallel.py` on windows machine to control parallelism.

Classification of 37 different types of dogs and cats:
| epoch    | train_loss  | valid_loss | error_rate | time  |
| ---------|:-----------:|:----------:|:----------:|:-----:|
| 0        | 0.947211    | 0.465167   | 0.142760   | 01:28 | 
| 1        | 0.577452    | 0.302070   | 0.085927   | 01:26 |
| 2        | 0.375799    | 0.253844   | 0.077808   | 01:25 |
| 3        | 0.294148    | 0.246899   | 0.074425   | 01:24 |

# Dependency installation - Env creation
I strongly recommend installing all the dependencies in a separate environment.  
`conda create -n fastai python=3.6`  
`conda activate fastai`  

When done just run
`conda deactivate`

To list all known environments:
`conda env list`  

# Dependency installation
`conda install pytorch torchvision cudatoolkit=10.2 -c pytorch`  
`conda install -c fastai fastai`  

# Fixing fastai for Windows
In order to be able to display images (show_batch did not display anything) in fastai I modified code
`~\anaconda3\Lib\site-packages\fastai\vision\data.py`  

by adding `plt.show()` at the end of the `show_xys` method call  
  
method after modification:  
```python
def show_xys(self, xs, ys, imgsize:int=4, figsize:Optional[Tuple[int,int]]=None, **kwargs):
        "Show the `xs` (inputs) and `ys` (targets) on a figure of `figsize`."
        rows = int(np.ceil(math.sqrt(len(xs))))
        axs = subplots(rows, rows, imgsize=imgsize, figsize=figsize)
        for x,y,ax in zip(xs, ys, axs.flatten()): x.show(ax=ax, y=y, **kwargs)
        for ax in axs.flatten()[len(xs):]: ax.axis('off')
        plt.tight_layout()
        plt.show()
```