# Data layout

Expected layout:

```text
data/
  baby/
    baby.inter
    image_feat.npy
    text_feat.npy
    user_emb.npy
  sports/
    sports.inter
    image_feat.npy
    text_feat.npy
    user_emb.npy
  elec/
    elec.inter
    image_feat.npy
    text_feat.npy
    user_emb.npy
```

Each `*.inter` file must include:

- user id column
- item id column
- split label column (`0` train, `1` validation, `2` test)

