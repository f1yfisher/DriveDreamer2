<div align="center">   

# DriveDreamer-2: LLM-Enhanced World Models for Diverse Driving Video Generation
</div>

Our team is actively working towards releasing the code for this project. 

We appreciate your patience and understanding as we navigate the necessary processes.
 
## [Project Page](https://drivedreamer2.github.io) | [Paper](https://arxiv.org/pdf/2403.06845.pdf)

# Abstract 

World models have demonstrated superiority in autonomous driving, particularly in the generation of multi-view driving videos. However, significant challenges still exist in generating customized driving videos. In this paper, we propose DriveDreamer-2, which builds upon the framework of DriveDreamer and incorporates a Large Language Model (LLM) to generate user-defined driving videos. Specifically, an LLM interface is initially incorporated to convert a user's query into agent trajectories. Subsequently, a HDMap, adhering to traffic regulations, is generated based on the trajectories. Ultimately, we propose the  Unified Multi-View Model to enhance temporal and spatial coherence in the generated driving videos. DriveDreamer-2 is the first world model to generate customized driving videos, it can generate uncommon driving videos (e.g., vehicles abruptly cut in) in a user-friendly manner. Besides, experimental results demonstrate that the generated videos enhance the training of driving perception methods (e.g., 3D detection and tracking). Furthermore, video generation quality of DriveDreamer-2 surpasses other state-of-the-art methods, showcasing FID and FVD scores of 11.2 and 55.7, representing relative improvements of 30% and 50%.

<img width="919" alt="abs" src="https://github.com/f1yfisher/DriveDreamer2/assets/39218234/e23cf401-5943-4fb3-b0ed-7d183a9df5cd">

<img width="1327" alt="abs2" src="https://github.com/f1yfisher/DriveDreamer2/assets/39218234/edc11963-0443-4e3f-8309-8955330b4815">




# News
- **[2024/03/11]** Repository Initialization.


# Demo
## Results with Gnerated Structural Information
**Daytime / rainy day / at night, a car abruptly cutting in from the right rear of ego-car.**

<div align="center">   

https://github.com/f1yfisher/DriveDreamer2/assets/39218234/0df78173-9dcd-42f4-8cf8-f7e16b724f82

</div>

**Rainy day, car abruptly cutting in from the left rear of ego-car. (long video)**

<div align="center">   

https://github.com/f1yfisher/DriveDreamer2/assets/39218234/779fa0ad-595a-47f3-a52c-1c98c30fa640

</div>

**Daytime, the ego-car changes lanes to the right side. (long video)**

<div align="center">   

https://github.com/f1yfisher/DriveDreamer2/assets/39218234/36c0f9e6-b9d1-4bd1-ab5c-f2c28eb3294c

</div>

**Rainy day, a person crosses the road in the front of the ego-car. (long video)**

<div align="center">   



https://github.com/f1yfisher/DriveDreamer2/assets/39218234/92f8cd31-a1b3-4516-ad03-331cf1ba4acb



</div>

## Results with nuScenes Structural Information

**Daytime / rainy day / at night, ego-car drives through urban street, surrounded by a flow of vehicles on both sides.**

<div align="center">   

https://github.com/f1yfisher/DriveDreamer2/assets/39218234/543656a4-729d-4b2c-b12d-6e75b3068669

</div>

**Daytime / rainy day / at night, a bus is positioned to the left front of the ego-car, with a pedestrian near the bus.**

<div align="center">   

https://github.com/f1yfisher/DriveDreamer2/assets/39218234/e43193ec-fb91-49ee-818c-b7a2c1a00909

</div>

**Rainy day, the windshield wipers of the truck are continuously clearing the windshield.**

<div align="center">   

https://github.com/f1yfisher/DriveDreamer2/assets/39218234/d05c2ab9-5c41-4dd3-bbd2-7a69b049b891

</div>

**Rainy day, the ego-car makes a left turn at the traffic signal, with vehicles behind proceeding straight through the intersection. (long video)**

<div align="center">   

https://github.com/f1yfisher/DriveDreamer2/assets/39218234/a766b12b-05a3-4755-858e-040c8bbf6ece

</div>

**Daytime, the ego-car drives straight through the traffic light, with a truck situated to the left front and pedestrians crossing on the right side. (long video)**

<div align="center">   

https://github.com/f1yfisher/DriveDreamer2/assets/39218234/e5f713dc-665f-49e2-8f70-3c5de101ffb4

</div>



# DriveDreamer-2 Framework

<img width="1277" alt="method" src="https://github.com/f1yfisher/DriveDreamer2/assets/39218234/bbb8d658-793a-4b3c-b873-ea5332f7ec4b">



# Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@article{zhao2024drive,
  title={DriveDreamer-2: LLM-Enhanced World Models for Diverse Driving Video Generation},
  author={Zhao, Guosheng and Wang, Xiaofeng and Zhu, Zheng and Chen, Xinze and Huang, Guan and Bao, Xiaoyi and Wang, Xingang},
  journal={arXiv preprint arXiv:2403.06845},
  year={2024}
}
```

