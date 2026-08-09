# Folder model

Letakkan berkas bobot hasil pelatihan di sini dengan nama:

```
EfficientNetCBAMV3.pth
```

Berkas ini adalah `state_dict` PyTorch dari kelas `EfficientNetB0_CBAM`
dengan 5 kelas keluaran, yaitu hasil dari:

```python
torch.save(model.state_dict(), SAVE_PATH)
```

Nama berkas bisa diubah lewat `DEFAULT_MODEL_PATH` di `config/settings.py`.

Berkas `.pth` sengaja dikecualikan dari Git lewat `.gitignore` karena
ukurannya besar.
