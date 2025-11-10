from sys import stderr
import json
import datetime
from sqlite3 import Connection, connect, register_adapter
from requests_sse import EventSource, InvalidStatusCodeError, InvalidContentTypeError
from requests import RequestException
import pandas as pd
from typing import Callable, List, Dict, Any

register_adapter(list, lambda x: json.dumps(x, ensure_ascii=False))

HEADERS = {"User-Agent": "Wikilacra/0.1 (https://jtlaune.github.io; jtlaune@gmail.com)"}

RECENTTAGS_COLS = [
    "schema",
    "database",
    "page_id",
    "page_title",
    "page_namespace",
    "rev_id",
    "rev_timestamp",
    "rev_sha1",
    "rev_minor_edit",
    "rev_len",
    "rev_content_model",
    "rev_content_format",
    "page_is_redirect",
    "comment",
    "parsedcomment",
    "rev_parent_id",
    "tags",
    "meta_uri",
    "meta_request_id",
    "meta_id",
    "meta_domain",
    "meta_stream",
    "meta_dt",
    "meta_topic",
    "meta_partition",
    "meta_offset",
    "prior_state_tags",
]

RECENTCHANGE_COLS = [
    "schema",
    "id",
    "type",
    "namespace",
    "title",
    "title_url",
    "comment",
    "timestamp",
    "user",
    "bot",
    "notify_url",
    "minor",
    "server_url",
    "server_name",
    "server_script_path",
    "wiki",
    "parsedcomment",
    "meta_uri",
    "meta_request_id",
    "meta_id",
    "meta_domain",
    "meta_stream",
    "meta_dt",
    "meta_topic",
    "meta_partition",
    "meta_offset",
    "length_old",
    "length_new",
    "revision_old",
    "revision_new",
]

REVISIONCREATE_COLS = [
    
]

def flush_batch(
    connection: Connection,
    batch: List[Dict[str, Any]],
    table: str,
    columns: List[str],
) -> int:
    if not batch:
        return 0
    df = pd.DataFrame(batch, columns=columns)
    df.to_sql(table, connection, if_exists="append", index=False)
    return len(batch)


def recenttag_filter(data: Dict[str, Any]) -> bool:
    meta = data.get("meta")
    return meta.get("domain") == "en.wikipedia.org" and data.get("page_namespace") == 0


def recentchange_filter(data: Dict[str, Any]) -> bool:
    meta = data.get("meta")
    return (
        meta.get("domain") == "en.wikipedia.org"
        and not data.get("bot")
        and data.get("namespace") == 0
        and (data.get("type") == "edit" or data.get("type") == "new")
    )


def flatten_json(keys: List[str], data: Dict[str, Any]) -> Dict[str, Any]:
    # "Flatten" data by popping out meta subdict, prefixing its
    # keys, and adding them to data.
    for k in keys:
        _ = {f"{k}_" + key: val for key, val in data.pop(k).items()}
        data = data | _
    return data


def stream_listen(
    endpoint: str,
    output_db_path: str,
    filter_func: Callable[[Dict[str, Any]], bool],
    flatten_keys: List[str],
    drop_keys: List[str],
    columns: List[str],
    table: str,
    batch_size=50,
) -> int:
    """Main function to listen to SSE stream from /v2/stream/{endpoint}. Flushes
    in batches of 50 to a duckdb connection at outputdb_path. Typically
    endpoint=recentchange or mediawiki.revision-tags-change."""
    con = connect(output_db_path)
    with EventSource(
        endpoint,
        timeout=30,
        headers=HEADERS,
    ) as event_source:
        batch = []
        batch_num = 0
        try:
            for event in event_source:
                try:
                    data = json.loads(event.data)
                except json.JSONDecodeError:
                    continue

                if filter_func(data):
                    for key in drop_keys:
                        if key in data.keys():
                            data.pop(key)
                    data = flatten_json(flatten_keys, data)
                    batch += [data]
                    if len(batch) >= batch_size:
                        flush_batch(con, batch, table, columns)
                        batch = []
                        batch_num += 1
                        print(f"batch={batch_num}", end="\r")

        except InvalidStatusCodeError:
            # pass
            return 1
        except InvalidContentTypeError:
            # pass
            return 2
        except RequestException:
            # pass
            return 3
        finally:
            # Final flush if anything left
            if batch:
                flush_batch(con, batch, table, columns)
                batch.clear()
            con.close()
        return 0


def create_tables(filename: str) -> None:
    con = connect(filename)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS recentchange (
         schema TEXT,
         id INT,
         type TEXT,
         namespace INT,
         title TEXT,
         title_url TEXT,
         comment TEXT,
         timestamp TEXT,
         user TEXT,
         bot TEXT,
         notify_url TEXT,
         minor TEXT,
         server_url TEXT,
         server_name TEXT,
         server_script_path TEXT,
         wiki TEXT,
         parsedcomment TEXT,
         meta_uri TEXT,
         meta_request_id TEXT,
         meta_id TEXT,
         meta_domain TEXT,
         meta_stream TEXT,
         meta_dt TEXT,
         meta_topic TEXT,
         meta_partition INT,
         meta_offset INT,
         length_old INT,
         length_new INT,
         revision_old INT,
         revision_new INT
        );
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS recenttags (
         schema TEXT,
         database TEXT,
         page_id INT,
         page_title TEXT,
         page_namespace INT,
         rev_id INT,
         rev_timestamp TEXT,
         rev_sha1 TEXT,
         rev_minor_edit TEXT,
         rev_len INT,
         rev_content_model TEXT,
         rev_content_format TEXT,
         page_is_redirect TEXT,
         comment TEXT,
         parsedcomment TEXT,
         rev_parent_id INT,
         tags TEXT,
         meta_uri TEXT,
         meta_request_id TEXT,
         meta_id TEXT,
         meta_domain TEXT,
         meta_stream TEXT,
         meta_dt TEXT,
         meta_topic TEXT,
         meta_partition INT,
         meta_offset INT,
         prior_state_tags TEXT
        );
        """
    )
    con.commit()
    con.close()
