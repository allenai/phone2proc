# 📱👉🏠 Phone2Proc

![my-file](https://github.com/allenai/phone2proc/assets/28768645/b5dfaec4-3a4b-45bd-bbaf-69fd379faf64)


This repository contains the code for [Phone2Proc](https://arxiv.org/abs/2212.04819). It performs conditional procedural generation to build a dataset of houses based on a specified [RoomPlan USDZ file](https://developer.apple.com/augmented-reality/roomplan/). The backend uses AI2-THOR and ProcTHOR. The generated houses are in the ProcTHOR house JSON format (see [this tutorial](https://colab.research.google.com/drive/1Il6TqmRXOkzYMIEaOU9e4-uTDTIb5Q78) on loading them into ProcTHOR). 

To obtain RoomPlan USDZ files, you'll need to load a custom app onto your phone, scan your scene, and then export it to a USDZ.

See the [this tutorial](https://www.youtube.com/watch?v=wgqwrgNiA68) and the [RoomPlan tutorial](https://developer.apple.com/documentation/roomplan/create_a_3d_model_of_an_interior_room_by_guiding_the_user_through_an_ar_experience) for more help.

To build the houses, install the requirements:

```bash
pip install -r requirements.txt
pip install --extra-index-url https://ai2thor-pypi.allenai.org ai2thor==0+391b3fae4d4cc026f1522e5acf60953560235971
```

Then run:

```bash
python3 dmain_wrapper.py
```

It'll generate houses based on the specified RoomPlan layout. You may need to go into `main.py` and change some constants, such as which USDZ file is being used.

Note that the RoomPlan scene must have all walls form an enclosed loop. Otherwise the generation will not work as it'll be underspecified how to generate the floorplan.

## 📖 Citation

To cite Phone2Proc, please use the following entry:

```bash
@inproceedings{phone2proc,
  title={Phone2proc: Bringing robust robots into our chaotic world},
  author={Deitke, Matt and Hendrix, Rose and Farhadi, Ali and Ehsani, Kiana and Kembhavi, Aniruddha},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9665--9675},
  year={2023}
}
```

## 👋 Our Team

AI2-THOR is an open-source project built by the [PRIOR team](//prior.allenai.org) at the [Allen Institute for AI](//allenai.org) (AI2).
AI2 is a non-profit institute with the mission to contribute to humanity through high-impact AI research and engineering.

<br />

<a href="//prior.allenai.org">
<p align="center"><img width="100%" src="https://raw.githubusercontent.com/allenai/ai2thor/main/doc/static/ai2-prior.svg" /></p>
</a>

