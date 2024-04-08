# CODIGEM - A Denoising Diffusion Probabilistic Model (DDPM) for Collaborative Filtering

How to execute the CODIGEM model

1. Download the dataset. For instance, Movielens-20m can be downloaded by following this link: https://grouplens.org/datasets/movielens/20m/

2. In the "data_processing.py" file, specify the required path named" path". 

3. In the "main.py" file, specify the required paths such as: "path" and "final_results_dir."

4. Run the code in any terminal. For instance, on your Anaconda terminal, the command would be: "python main.py"

```Python
   python main.py
```

5. Check the results in the folders earlier specified.

## Abstract 
Despite the success of classical collaborative filtering (CF) methods in the recommendation systems domain, we point out two issues that essentially limit this class of models. Firstly, most classical CF models predominantly yield weak collaborative signals, which makes them deliver suboptimal recommendation performance. Secondly, most classical CF models produce unsatisfactory latent representations resulting in poor model generalization and performance. To address these limitations, this paper presents the Collaborative Diffusion Generative Model (CODIGEM), the _first-ever_ denoising diffusion probabilistic model (DDPM)-based CF model. CODIGEM effectively models user-item interactions data by obtaining the intricate and non-linear patterns to generate strong collaborative signals and robust latent representations for improving the model’s generalizability and recommendation performance. Empirically, we demonstrate that CODIGEM is a very efficient generative CF model, and it outperforms several classical CF models on several real-world datasets. Moreover, we illustrate through experimental validation the settings that make CODIGEM provide the most significant recommendation performance, highlighting the importance of using the DDPM in recommendation systems.

**Please cite this paper if you use our code. The bibtex is presented below**

```
@inproceedings{DBLP:conf/ksem/WalkerZZG022,
  author       = {Joojo Walker and
                  Ting Zhong and
                  Fengli Zhang and
                  Qiang Gao and
                  Fan Zhou},
  title        = {Recommendation via Collaborative Diffusion Generative Model},
  booktitle    = {Knowledge Science, Engineering and Management - 15th International
                  Conference, {KSEM} 2022, Singapore, August 6-8, 2022, Proceedings,
                  Part {III}},
  series       = {Lecture Notes in Computer Science},
  volume       = {13370},
  pages        = {593--605},
  publisher    = {Springer},
  year         = {2022},
  url          = {https://doi.org/10.1007/978-3-031-10989-8\_47},
  doi          = {10.1007/978-3-031-10989-8\_47}
}
```
