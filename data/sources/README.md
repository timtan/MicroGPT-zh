# 台灣姓名彙總資料來源

這個目錄的 CSV 由內政部戶政司《112 年全國姓名統計分析》抽取，統計基準日為民國 112 年 6 月 30 日。

- 原始 PDF：https://www.ris.gov.tw/documents/data/5/2/112namestat.pdf
- 政府資料開放授權條款第 1 版：https://data.gov.tw/license
- `taiwan_given_names_112.csv`：表五十一，男女各前 100 大常見名字。
- `taiwan_common_full_names_112.csv`：表五十二，男女各前 100 大同姓同名；只作 holdout 檢查，不加入訓練。
- `taiwan_surnames_112.csv`：表五十七，姓氏排名與總人口數。

重建方式：

```bash
pdftotext -layout 112namestat.pdf 112namestat.txt
uv run python scripts/extract_taiwan_name_tables.py 112namestat.txt
```

一般姓名候選另外來自 `wainshine/Chinese-Names-Corpus`，版本
`47d4af8d816f6212787ddfc49173cac3b994b58d`，依 Apache License 2.0 使用。原始大型語料不納入本儲存庫；建置腳本只保留經台灣官方彙總資料校準後的衍生姓名集。
