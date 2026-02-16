# RoboKudo
RoboKudo is an open source framework for robot perception for ROS2

## Installation instructions for Ubuntu (tested on 24.04)

### (Optional) create a virtual environment using virtualenvwrapper
You can use the same virtual environment as the one that CRAM uses. 
If you want to use a different one, you can create one using virtualenvwrapper.

```
sudo apt install virtualenvwrapper
echo "export WORKON_HOME=~/venvs" >> ~/.bashrc
echo "source /usr/share/virtualenvwrapper/virtualenvwrapper.sh" >> ~/.bashrc
source ~/.bashrc
mkdir -p $WORKON_HOME

# --system-site-packages is only required if you are using ROS
mkvirtualenv robokudo --system-site-packages
```
To use it, call:
```
workon robokudo
```

### Install and use RoboKudo
- Clone the CRAM repository to your filesystem. In this example, we'll use ~/libs: 
```
mkdir -p ~/libs && cd ~/libs
git clone https://github.com/cram2/cognitive_robot_abstract_machine.git
cd robokudo
```
- Switch to your venv, if you use one.
```
workon robokudo
```
- Install Giskardpy, `-e` is optional but prevents you from having to rebuild every time the code changes.
```
pip3 install -r requirements.txt
pip3 install -e .                           
```

### Tutorials
https://robokudo.ai.uni-bremen.de/

### How to cite
```
@inproceedings{mania2024robokudo,
	title={An Open and Flexible Robot Perception Framework for Mobile Manipulation Tasks},
	author={Mania, Patrick and Stelter, Simon and Kazhoyan, Gayane and Beetz, Michael},
	booktitle={2024 International Conference on Robotics and Automation (ICRA)},
	year={2024},
	url={https://ai.uni-bremen.de/papers/mania2024robokudo.pdf},
	note={},
	organization={IEEE}
}
```
