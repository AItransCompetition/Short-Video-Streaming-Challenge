# Submit Demo
* The type of file you submit is *.zip. the name is like submit.zip
* The submit.zip is gotten by compressed a folder called submit.The path is like following:
```
Submit
│   README.md
│   solution.py    
└───results
│   │   your_model.pb
│   │   your_other_file

```

# solution.py
* PATH
if you want to call your model ,the path setting is 
```
NN_MODEL = "/home/team/"$YOUR TEAM NAME"/submit/results/nn_model_ep_18200.ckpt" # model path settings
```
* Algorithm

* - from simulator.video_player import Player

  You will be returned an instance of the Player class with every decision step to aid your next decision. 

  Please read video_player.py for details.

* * Init: you can init some self.params 
```
 def __init__(self):
     # fill your init vars
         self.buffer_size = 0
```
* * Initial your params: 
```
# Intialize
     def Initialize(self):
             IntialVars = []
             return IntialVars

```
* * Run: your algorithm logic
```
def run(self, delay, rebuf, video_size, end_of_video, play_video_id, Players):
    
         download_video_id = 0
         bit_rate = 0
         sleep_time = 0.0
         return download_video_id, bit_rate, sleep_time
```

# Import package
* The official submission system has provided some pre-installed libraries. The following is the pre-installed library name and version number information

  | Library name | numpy  | torch  | tensorflow |
  | :----------: | :----: | :----: | :--------: |
  |   Version    | 1.21.5 | 1.11.0 |   2.8.0    |

* If you want to add some site package ,please concact zuoxt18@mails.tsinghua.edu.cn

