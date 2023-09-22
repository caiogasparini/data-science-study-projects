import pandas as pd
from pandas.io.parsers.readers import *
import csv
import polars as pl
from typing import Literal

## same imports in pandas.io.parsers.readers for pd_read_csv params
from typing import (
    TYPE_CHECKING,
    Callable,
    Literal,
)

from pandas._libs import lib

from collections.abc import (
    Hashable,
    Mapping,
    Sequence,
)

from pandas._typing import (
    CompressionOptions,
    CSVEngine,
    DtypeArg,
    DtypeBackend,
    FilePath,
    HashableT,
    IndexLabel,
    ReadCsvBuffer,
    StorageOptions,
)
## end

## same imports in pandas.io.parsers.readers for pl_read_csv params
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Callable, Mapping, Sequence, TextIO
from polars.datatypes import N_INFER_DEFAULT

from io import BytesIO

from polars.type_aliases import CsvEncoding, PolarsDataType, SchemaDict
## end 

class DataFrameOptimizer:
    def __init__(self) -> None:
        pass

    @classmethod
    def __mapping_dtype_groups(cls, df: pl.DataFrame) -> dict:
        dtype_groups: dict[str, list] = {
            'Int': [],
            'Float': [],
            'Object': []
        }
        for c in df.columns:
            pl_dtype: pl.PolarsDataType = df[c].dtype
            match pl_dtype:
                case pl.Int64:
                    dtype_groups['Int'].append(c)
                case pl.Float64:
                    dtype_groups['Float'].append(c)
                case pl.Object:
                    dtype_groups['Object'].append(c)
        return dtype_groups

    @classmethod
    def __df_converter(cls, dataframe: (pl.DataFrame | pd.DataFrame)) -> (pl.DataFrame | pd.DataFrame):
        df: (pl.DataFrame | pd.DataFrame) = dataframe
        if type(df) is pd.DataFrame:
            df = pl.from_pandas(df)
        else:
            df = df.to_pandas()
        return df

    @classmethod
    def __downcasting(cls, dataframe: (pl.DataFrame | pd.DataFrame), dtype_groups: dict[str, list]) -> dict:
        df: (pl.DataFrame | pd.DataFrame) = dataframe
        if type(df) == pl.DataFrame:
            df: pd.DataFrame = cls.__df_converter(dataframe)
        for c in dtype_groups['Int']:
            df[c] = pd.to_numeric(df[c], downcast='integer')
        for c in dtype_groups['Float']:
            df[c] = pd.to_numeric(df[c], downcast='float')
        for c in dtype_groups['Object']:
            df[c] = pd.Categorical(df[c]) # df[c].astype('category')
        column_types_pd: dict = dict(zip(df.columns, df.dtypes))
        return column_types_pd       

    @classmethod
    def __pd_types_conversor(self, column_types_pd) -> dict:
        pl_num_dtypes: dict = {
            'int8': pl.Int8,
            'int16': pl.Int16,
            'int32': pl.Int32,
            'int64': pl.Int64,
            'float32': pl.Int32,
            'float64': pl.Int64
        }
        column_types_pl: dict = {}
        for index, value in column_types_pd.items():  
            if value in pl_num_dtypes:
                column_types_pl[index] = pl_num_dtypes[value]
            else:
                column_types_pl[index] = pl.Categorical
        return column_types_pl

    @classmethod
    def __get_improvement(cls, dataframe: (pl.DataFrame | pd.DataFrame), file_path: str, column_types_pd: dict):
        df: (pl.DataFrame | pd.DataFrame) = dataframe
        if type(df) == pl.DataFrame:
            df: pd.DataFrame = cls.__df_converter(dataframe)
        standard_memory = df.memory_usage().sum()
        improvement_memory: (pd.DataFrame | any) = pd.read_csv(file_path, nrows=1000, dtype=column_types_pd)
        improvement_memory = improvement_memory.memory_usage().sum()
        return print(f'Estimated saved memory -> {(improvement_memory / standard_memory) * 100:.2f}%')

    # read_csv decorator
    def __dec_read_csv(func) -> (pd.DataFrame | pl.DataFrame):
        def main_func(*args, **kwargs)  -> (pd.DataFrame | pl.DataFrame):
            # inicializando vars
            f_path: str = ''
            df: pl.DataFrame | pd.DataFrame = pl.DataFrame()
            func_name: str = func.__name__
            new_args: tuple # se der erro pode ser necessário transformar em list
            if (func_name == 'read_csv') & (len(args) > 2):
                f_path = args[2]
                new_args = args[2:] # remover self e select_lib do args
            elif (func_name != 'read_csv') & (len(args) > 1):
                f_path = args[1]
                new_args = args[1:] # remover self do args
            else:
                f_path = kwargs['file_path']
                new_args = ()
        
            if func_name == 'pd_read_csv':
                df = pd.read_csv(*new_args, **kwargs)
                df = DataFrameOptimizer.__df_converter(df)
            else:
                df = pl.read_csv(*new_args, **kwargs)
            dtype_groups: dict = DataFrameOptimizer.__mapping_dtype_groups(df)
            column_types_pd: dict = DataFrameOptimizer.__downcasting(df, dtype_groups)
            DataFrameOptimizer.__get_improvement(df, f_path, column_types_pd)

            # guardando valores em kwargs para repassar para a função
            print(column_types_pd)
            if kwargs.__contains__('select_lib'):
                if func_name == 'pl_read_csv' or args[1] == 'polars' or kwargs['select_lib'] == 'polars':
                    kwargs['dtypes'] = DataFrameOptimizer.__pd_types_conversor(column_types_pd)
                else:
                    kwargs['dtype'] = column_types_pd
            if func_name == 'read_csv':
                return func(args[0], args[1], args[2], *new_args, **kwargs)
            else:
                return func(args[0], args[1], *new_args, **kwargs)
        return main_func

    @__dec_read_csv
    def read_csv(self,select_lib: Literal["polars", "pandas"], file_path: str,*args , **kwargs) -> (pd.DataFrame | pl.DataFrame):
        # independente dos dtypes do pandas, o polars importará pelo modelo padrão int64/float64.
        if select_lib == "pandas":
            return pd.read_csv(*args, **kwargs)
        elif select_lib == "polars":
            return pl.read_csv(*args, **kwargs)
        
    @__dec_read_csv    
    def pd_read_csv(
        self,
        filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
        *args, # alterado de * para *args
        sep: str | None | lib.NoDefault = lib.no_default,
        delimiter: str | None | lib.NoDefault = None,
        # Column and Index Locations and Names
        header: int | Sequence[int] | None | Literal["infer"] = "infer",
        names: Sequence[Hashable] | None | lib.NoDefault = lib.no_default,
        index_col: IndexLabel | Literal[False] | None = None,
        usecols: list[HashableT] | Callable[[Hashable], bool] | None = None,
        # General Parsing Configuration
        dtype: DtypeArg | None = None,
        engine: CSVEngine | None = None,
        converters: Mapping[Hashable, Callable] | None = None,
        true_values: list | None = None,
        false_values: list | None = None,
        skipinitialspace: bool = False,
        skiprows: list[int] | int | Callable[[Hashable], bool] | None = None,
        skipfooter: int = 0,
        nrows: int = 1000,
        # NA and Missing Data Handling
        na_values: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
        keep_default_na: bool = True,
        na_filter: bool = True,
        verbose: bool = False,
        skip_blank_lines: bool = True,
        # Datetime Handling
        parse_dates: bool | Sequence[Hashable] | None = None,
        infer_datetime_format: bool | lib.NoDefault = lib.no_default,
        keep_date_col: bool = False,
        date_parser: Callable | lib.NoDefault = lib.no_default,
        date_format: str | None = None,
        dayfirst: bool = False,
        cache_dates: bool = True,
        # Iteration
        iterator: bool = False,
        chunksize: int | None = None,
        # Quoting, Compression, and File Format
        compression: CompressionOptions = "infer",
        thousands: str | None = None,
        decimal: str = ".",
        lineterminator: str | None = None,
        quotechar: str = '"',
        quoting: int = csv.QUOTE_MINIMAL,
        doublequote: bool = True,
        escapechar: str | None = None,
        comment: str | None = None,
        encoding: str | None = None,
        encoding_errors: str | None = "strict",
        dialect: str | csv.Dialect | None = None,
        # Error Handling
        on_bad_lines: str = "error",
        # Internal
        delim_whitespace: bool = False,
        low_memory: bool = True,
        memory_map: bool = False,
        float_precision: Literal["high", "legacy"] | None = None,
        storage_options: StorageOptions | None = None,
        dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
        **kwargs
    ) -> pd.DataFrame:
        return pd.read_csv(*args, **kwargs)

    @__dec_read_csv
    def pl_read_csv(
        self,
        source: str | TextIO | BytesIO | Path | BinaryIO | bytes,
        *args, # alterado de * para *args
        has_header: bool = True,
        columns: Sequence[int] | Sequence[str] | None = None,
        new_columns: Sequence[str] | None = None,
        separator: str = ",",
        comment_char: str | None = None,
        quote_char: str | None = r'"',
        skip_rows: int = 0,
        dtypes: Mapping[str, PolarsDataType] | Sequence[PolarsDataType] | None = None,
        schema: SchemaDict | None = None,
        null_values: str | Sequence[str] | dict[str, str] | None = None,
        missing_utf8_is_empty_string: bool = False,
        ignore_errors: bool = False,
        try_parse_dates: bool = False,
        n_threads: int | None = None,
        infer_schema_length: int | None = N_INFER_DEFAULT,
        batch_size: int = 8192,
        n_rows: int = 1000, # alterado para manter por padrão uma amostra de 1000 linhas
        encoding: CsvEncoding | str = "utf8",
        low_memory: bool = False,
        rechunk: bool = True,
        use_pyarrow: bool = False,
        storage_options: dict[str, Any] | None = None,
        skip_rows_after_header: int = 0,
        row_count_name: str | None = None,
        row_count_offset: int = 0,
        sample_size: int = 1024,
        eol_char: str = "\n",
        raise_if_empty: bool = True,
        truncate_ragged_lines: bool = False,
        **kwargs
    ) -> pl.DataFrame:
        return pl.read_csv(*args, **kwargs)