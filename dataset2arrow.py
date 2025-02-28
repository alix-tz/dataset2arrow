from datasets import load_dataset, DatasetDict
from typing import Tuple, List
from PIL import Image
from collections import Counter
import logging
import tempfile
import pyarrow as pa
import numpy as np
import json
import io
import tabulate
import click
import re

from numpy.f2py.cfuncs import callbacks

logging.basicConfig()
logger = logging.getLogger(__name__)

def resize_image(row, max_height: int = 120):
    """Resize image to max height of 120 pixels while maintaining aspect ratio."""
    row["im"] = row['im'].convert("L")
    image = row['im']  # Assuming the image is stored in the 'im' column
    if image.height > max_height:
        aspect_ratio = image.width / image.height
        new_height = max_height
        new_width = int(aspect_ratio * new_height)
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        row['im'] = resized_image
    return row

def build_arrow(dataset: DatasetDict, output_file: str = "filtered.arrow", recordbatch_size: int = 100, text_col='text'):
    """ Builds an arrow for Kraken using a DatasetDict

    :param dataset: A dataset loaded through load_dataset from one of our repository
    :param output_file: The target file to save
    :param recordbatch_size: Don't touch it if you don't know what it is. See Kraken doc.

    """
    logger.info('Assembling dataset alphabet.')
    alphabet = Counter()
    num_lines = 0
    for split in dataset:
        for row in dataset[split][text_col]:
            alphabet.update(row)
            num_lines += 1

    logger.info("Creating structure")
    metadata = {
        'lines': {
            'type': 'kraken_recognition_baseline',
            'alphabet': alphabet,
            'text_type': 'raw',
            'image_type': 'raw',
            'splits': ['train', 'eval', 'test'],
            'im_mode': '1',
            'legacy_polygons': False,
            'counts': Counter({
                'all': num_lines,
                'train': dataset["train"].num_rows,
                'validation': dataset["validation"].num_rows,
                'test': dataset["test"].num_rows
            }),
        }
    }
    ty = pa.struct([('text', pa.string()), ('im', pa.binary())])
    schema = pa.schema([('lines', ty), ('train', pa.bool_()), ('validation', pa.bool_()), ('test', pa.bool_())])

    def _make_record_batch(line_cache, target: str):
        ar = pa.array(line_cache, type=ty)

        tr_ind = np.zeros(len(line_cache), dtype=bool)
        val_ind = np.zeros(len(line_cache), dtype=bool)
        test_ind = np.zeros(len(line_cache), dtype=bool)

        if target == "train":
            tr_ind = np.ones(len(line_cache), dtype=bool)
        elif target == "validation":
            val_ind = np.ones(len(line_cache), dtype=bool)
        elif target == "test":
            test_ind = np.ones(len(line_cache), dtype=bool)

        logger.debug(f"Split: {target}, Train: {tr_ind.sum()}, Validation: {val_ind.sum()}, Test: {test_ind.sum()}")

        train_mask = pa.array(tr_ind)
        val_mask = pa.array(val_ind)
        test_mask = pa.array(test_ind)
        rbatch = pa.RecordBatch.from_arrays([ar, train_mask, val_mask, test_mask], schema=schema)
        return rbatch, (len(line_cache), int(sum(tr_ind)), int(sum(val_ind)), int(sum(test_ind)))

    line_cache = []
    logger.info('Writing lines to temporary file.')
    with tempfile.TemporaryDirectory() as tmp_output_dir:
        tmp_file = tmp_output_dir + '/dataset.arrow'
        with pa.OSFile(tmp_file, 'wb') as sink:
            with pa.ipc.new_file(sink, schema) as writer:
                for split in dataset:
                    for row in dataset[split]:
                        fp = io.BytesIO()
                        row["im"].save(fp, format='png')
                        line_cache.append({"text": row[text_col], "im": fp.getvalue()})  # Check format of row
                        if len(line_cache) == recordbatch_size:
                            logger.debug(f'Flushing {len(line_cache)} lines into {tmp_file}.')
                            rbatch, counts = _make_record_batch(line_cache, target=split)
                            writer.write(rbatch)
                            line_cache = []

                    if len(line_cache):
                        rbatch, counts = _make_record_batch(line_cache, target=split)
                        writer.write(rbatch)
                        line_cache = []

        logger.info('Dataset metadata')
        logger.info(f"type: {metadata['lines']['type']}\n"
                    f"text_type: {metadata['lines']['text_type']}\n"
                    f"image_type: {metadata['lines']['image_type']}\n"
                    f"splits: {metadata['lines']['splits']}\n"
                    f"im_mode: {metadata['lines']['im_mode']}\n"
                    f"legacy_polygons: {metadata['lines']['legacy_polygons']}\n"
                    f"lines: {metadata['lines']['counts']}\n")

        with pa.memory_map(tmp_file, 'rb') as source:
            logger.info(f'Rewriting output ({output_file}) to update metadata.')
            ds = pa.ipc.open_file(source).read_all()
            metadata['lines']['counts'] = dict(metadata['lines']['counts'])
            metadata['lines'] = json.dumps(metadata['lines'])
            schema = schema.with_metadata(metadata)
            with pa.OSFile(output_file, 'wb') as sink:
                with pa.ipc.new_file(sink, schema) as writer:
                    writer.write(ds)
    return


def _parse_filters(ctx, param, values: Tuple[str, ...]) -> List[Tuple[str, str]]:
    out = []
    for value in values:
        if re.match(r'(\w+)=(\w+)', value):
            out.append(tuple(value.split("=")[:2]))
        else:
            raise click.BadParameter('.....')
    return out


def _parse_sample(ctx, param, values: Tuple[str, str]) -> Tuple[str, int]:
    try:
        return values[0], int(values[1])
    except:
        raise
        raise click.BadParameter('.....')

class Sampler:
    def __init__(self, column: str, max_value: int):
        self.column: str = column
        self.max_value: int = max_value
        self.counter: Counter = Counter()

    def __call__(self, row) -> bool:
        self.counter[row[self.column]] += 1
        return self.counter[row[self.column]] <= self.max_value

    def reset(self):
        self.counter = Counter()

    def pretty_print(self):
        print(
            tabulate.tabulate(
                [
                    [key, value, f"{100*(max(value-self.max_value, 0) / value):.1f}%"]
                    for key, value in self.counter.items()
                ],
                headers=["Value", "Found", "Discarded"],
                tablefmt="md"
            )
        )


@click.command()
@click.argument("dataset")
@click.argument("output", type=click.Path(file_okay=True, dir_okay=False))
@click.option("-m", "--max-height", type=int, default=120)
@click.option("-f", "--filters", callback=_parse_filters, multiple=True, default=[])
@click.option("-v", "--verbose", default=False, is_flag=True)
@click.option("-s", "--sample", type=(str, int),
              help="`--sample shelfmark 1000` will ensure you don't have more "
                   "than 1000 rows with the same manuscript value per split")
@click.option("--col", type=str, default='text', help="--col specifies the name of "
                "the column to use to extract labels (default is 'text')")
@click.option("-l", "--local", default=False, is_flag=True, help="will load local parquet files (for dev purpose)")
def cli(dataset, output, max_height, filters, verbose, sample, local, col):#, datasplit):
    """ Convert [DATASET] into [OUTPUT], a Kraken arrow file. You can use filters such as
    `python dataset2arrow.py CATMuS/medieval-sample latin-9th-century.arrow --filters language=Latin --filters century=9`

    Use this commandline code as a documentation for your own modification.
    Using `--max-height 120` is recommended if you are using the default Kraken specs, as it lowers the file size
    Using `--sample shelfmark 100` does not use more than 100 lines from each selfmark. Samples are retrieved in order
    """
    if verbose:
        logger.setLevel(logging.INFO)

    if local:
        d: DatasetDict = load_dataset("parquet", data_dir=dataset)
    else:
        d: DatasetDict = load_dataset(dataset)

    for key, value in filters:
        if value.isnumeric():
            value = int(value)
        d = d.filter(lambda row: row[key] == value, batched=False)
    d = d.map(lambda x: resize_image(x, max_height))

    if sample:
        sampler = Sampler(sample[0], sample[1])
        for split in d:
            d[split] = d[split].filter(sampler)
            if verbose:
                sampler.pretty_print()
            sampler.reset()

    build_arrow(d, output_file=output, text_col=col)


if __name__ == "__main__":
    cli()