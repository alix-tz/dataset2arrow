# dataset2arrow

This small library allows you to compile a CATMuS Dataset into a Kraken *base* arrow, while filtering out datasets.

It provides small helper functions to compile into arrows based on the DatasetDict obtained through `load_dataset()` from the HuggingFace library. 

You can use it as a cli, with commands such as `python dataset2arrow.py CATMuS/medieval-sample latin.arrow --filters language=Latin` (You can use multiple filters)

Use this commandline code as a documentation for your own modification.

Using `--max-height 120` is recommended if you are using the default Kraken specs, as it lowers the file size