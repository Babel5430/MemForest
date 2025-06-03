import enum
import re
from typing import Any, Dict, Optional, List, Tuple, Union

try:
    from qdrant_client import models
    from qdrant_client.http.models import Filter, FieldCondition, Range, MatchValue, MatchText, MatchAny, MatchExcept
    QDRANT_AVAILABLE = True
except ImportError:
    print("Qdrant client (qdrant-client) not installed. Qdrant support will be disabled.")
    QDRANT_AVAILABLE = False
class FilterOperator(enum.Enum):
    EQ = "=="
    NE = "!="
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    IN = "in"
    NOT_IN = "not in"
    CONTAINS = "contains"
    LIKE = "like"

class BaseFilter:
    """Base class for all filter types."""
    def to_milvus_expr(self) -> Optional[str]:
        """Converts the filter to a Milvus/Milvus-lite expression string."""
        raise NotImplementedError

    def to_chroma_filter(self) -> Optional[Dict[str, Any]]:
        """Converts the filter to a ChromaDB metadata filter dictionary."""
        raise NotImplementedError

    def to_sqlite_where(self) -> Optional[str]:
        """Converts the filter to a SQLite WHERE clause string."""
        raise NotImplementedError

    def to_qdrant_filter(self):
        """Converts the filter to a Qdrant filters."""
        raise NotImplementedError

class FieldFilter(BaseFilter):
    """Represents a filter on a specific field."""
    def __init__(self, field: str, operator: FilterOperator, value: Any):
        if not isinstance(field, str) or not field:
            raise ValueError("Field name must be a non-empty string.")
        if not isinstance(operator, FilterOperator):
            raise ValueError("Operator must be a FilterOperator enum.")
        # Basic validation for value types based on operator could be added here
        self.field = field
        self.operator = operator
        self.value = value

    def to_milvus_expr(self) -> Optional[str]:
        """Converts FieldFilter to Milvus/Milvus-lite expression string."""
        def _escape_milvus_value(val: Any) -> Optional[str]:
            if isinstance(val, str):
                # Escape double quotes for Milvus string literals
                escaped_val = val.replace('"', '\\"')
                return '"{}"'.format(escaped_val)
            elif isinstance(val, (int, float, bool)):
                return str(val).lower() if isinstance(val, bool) else str(val)
            elif isinstance(val, (list, tuple)) and self.operator in [FilterOperator.IN, FilterOperator.NOT_IN]:
                 formatted_elements = []
                 for item in val:
                     if isinstance(item, (str, int, float, bool)):
                          formatted_elements.append(_escape_milvus_value(item))
                     else:
                          return None
                 return "[{}]".format(', '.join(e for e in formatted_elements if e is not None)) if all(e is not None for e in formatted_elements) else None
            elif val is None:
                 return 'null'
            return None

        formatted_value = _escape_milvus_value(self.value)
        if formatted_value is None:
            print(f"Warning: Cannot convert value {self.value} for field '{self.field}' to Milvus format.")
            return None

        milvus_op = self.operator.value
        if self.operator == FilterOperator.LIKE:
             # Milvus uses LIKE with single quotes, escape single quotes within value
             if isinstance(self.value, str):
                  escaped_value = self.value.replace("'", "''")
                  return f"{self.field} like '{escaped_value}'"
             else:
                  print(f"Warning: LIKE operator requires string value for field '{self.field}'.")
                  return None
        elif self.operator in [FilterOperator.IN, FilterOperator.NOT_IN]:
             if formatted_value.startswith('[') and formatted_value.endswith(']'):
                  return f"{self.field} {milvus_op} {formatted_value}"
             else:
                  print(f"Warning: Value for IN/NOT IN operator must be a list/tuple for field '{self.field}'.")
                  return None
        elif self.operator == FilterOperator.CONTAINS:
             # Milvus doesn't have a direct 'contains' for scalar fields.
             # For JSON fields, it might support `json_contains`.
             # Leave for latter.
             print(f"Warning: CONTAINS operator not directly supported for scalar fields like '{self.field}' in Milvus.")
             return None

        # Default comparison operators (==, !=, >, >=, <, <=)
        return f"{self.field} {milvus_op} {formatted_value}"


    def to_chroma_filter(self) -> Optional[Dict[str, Any]]:
        """Converts FieldFilter to ChromaDB metadata filter dictionary."""
        chroma_op_map = {
            FilterOperator.EQ: '$eq', FilterOperator.NE: '$ne',
            FilterOperator.GT: '$gt', FilterOperator.GTE: '$gte',
            FilterOperator.LT: '$lt', FilterOperator.LTE: '$lte',
            FilterOperator.IN: '$in', FilterOperator.NOT_IN: '$nin',
            FilterOperator.CONTAINS: '$contains', # Chroma has $contains for list/string
            # Chroma doesn't have a direct 'like'.
            FilterOperator.LIKE: '$contains'
        }
        chroma_op = chroma_op_map.get(self.operator)
        if not chroma_op:
             print(f"Warning: Unsupported Chroma operator conversion for {self.operator}.")
             return None

        # Convert field name from Python access (e.g., metadata["key"]) to Chroma dot notation (e.g., metadata.key)
        chroma_field = self.field
        # Simplified regex for common metadata["key"] or metadata['key'] pattern
        match = re.match(r"([a-zA-Z0-9_]+)\[[\"']?([a-zA-Z0-9_]+)[\"']?\]", self.field)
        if match:
            field_name, key_name = match.groups()
            # Assuming this syntax is only used for 'metadata' field in MemoryUnit
            if field_name == 'metadata':
                 # print("Warning: chroma do not support json extract currently.")
                 # return None
                 chroma_field = key_name
            else:
                 print(f"Warning: Field access syntax '{self.field}' used for non-'metadata' field. Using raw field name for Chroma.")
                 chroma_field = self.field # Fallback to raw field name
        elif '[' in self.field or ']' in self.field:
             print(f"Warning: Complex field access '{self.field}' may not convert correctly to Chroma dot notation. Using raw field name.")
             chroma_field = self.field # Fallback to raw field name

        # Value formatting for Chroma filters:
        formatted_value = self.value

        if self.operator in [FilterOperator.IN, FilterOperator.NOT_IN]:
             if not isinstance(self.value, (list, tuple)):
                  print(f"Warning: Value for IN/NOT IN operator must be a list/tuple for field '{self.field}' for Chroma.")
                  return None
             # Ensure list elements are simple types compatible with Chroma
             if not all(isinstance(item, (str, int, float, bool)) for item in self.value):
                   print(f"Warning: List elements for IN/NOT IN must be simple types for field '{self.field}' for Chroma.")
                   return None
             # Chroma expects a list for $in/$nin
             formatted_value = list(self.value)
        elif self.operator == FilterOperator.LIKE:
             if not isinstance(self.value, str):
                  print(f"Warning: LIKE operator requires string value for field '{self.field}'.")
                  return None
             # If LIKE is approximated as $contains, the value is the substring.
             if '*' in self.value or '%' in self.value:
                   print(f"Warning: LIKE wildcard conversion to $contains is approximate for field '{self.field}'.")
             formatted_value = self.value # The substring to search for
        # Other operators (==, !=, >, >=, <, <=, contains) generally use the value directly

        # Construct the Chroma filter dictionary using the converted field name
        return {chroma_field: {chroma_op: formatted_value}}

    def to_qdrant_filter(self) -> Optional[models.Filter]:
        """Converts FieldFilter to a Qdrant Filter object."""
        if not QDRANT_AVAILABLE: return None

        field_name = self.field
        operator = self.operator
        value = self.value

        # Convert field name from Python access (e.g., metadata["key"]) to Qdrant dot notation (e.g., metadata.key)
        qdrant_field = field_name
        match = re.match(r"([a-zA-Z0-9_]+)\[[\"']?([a-zA-Z0-9_]+)[\"']?\]", self.field)
        if match:
            field_name_base, key_name = match.groups()
            qdrant_field = f"{field_name_base}.{key_name}"
        elif '.' in self.field:
             qdrant_field = self.field
        try:
            if operator == FilterOperator.EQ:
                condition = FieldCondition(key=qdrant_field, match=MatchValue(value=value))
            elif operator == FilterOperator.NE:
                 condition = FieldCondition(key=qdrant_field, match=MatchExcept(except_=MatchValue(value=value)))
            elif operator == FilterOperator.GT:
                condition = FieldCondition(key=qdrant_field, range=Range(gt=value))
            elif operator == FilterOperator.GTE:
                condition = FieldCondition(key=qdrant_field, range=Range(gte=value))
            elif operator == FilterOperator.LT:
                condition = FieldCondition(key=qdrant_field, range=Range(lt=value))
            elif operator == FilterOperator.LTE:
                condition = FieldCondition(key=qdrant_field, range=Range(lte=value))
            elif operator == FilterOperator.IN:
                if isinstance(value, (list, tuple)):
                    condition = FieldCondition(key=qdrant_field, match=MatchAny(any=list(value)))
                else:
                    print(f"Warning: Value for IN operator must be a list/tuple for field '{self.field}'.")
                    return None
            elif operator == FilterOperator.NOT_IN:
                if isinstance(value, (list, tuple)):
                     condition = FieldCondition(key=qdrant_field, match=MatchExcept(except_=MatchAny(any=list(value))))
                else:
                    print(f"Warning: Value for NOT_IN operator must be a list/tuple for field '{self.field}'.")
                    return None
            elif operator == FilterOperator.CONTAINS:
                 # Assuming it primarily means checking if a single 'value' is present in a list/array field.
                 condition = FieldCondition(key=qdrant_field, match=MatchAny(any=[value]))
                 # print(f"Warning: CONTAINS operator mapping to Qdrant MatchAny(any=[value]) for field '{self.field}'. Ensure the field is a list/array type in Qdrant.")
            elif operator == FilterOperator.LIKE:
                 if isinstance(value, str):
                      condition = FieldCondition(key=qdrant_field, match=MatchText(text=value))
                      # print(f"Warning: LIKE operator mapping to Qdrant MatchText for field '{self.field}'. Full wildcard support not guaranteed without specific text indexing.")
                 else:
                     print(f"Warning: LIKE operator requires string value for field '{self.field}'.")
                     return None
            else:
                print(f"Warning: Unsupported Qdrant operator conversion for {operator}.")
                return None

            if condition:
                return Filter(must=[condition])

        except Exception as e:
            print(f"Error creating Qdrant FieldCondition for field '{self.field}': {e}")
            return None

        return None

    def to_sqlite_where(self) -> Optional[str]:
        """Converts FieldFilter to SQLite WHERE clause string."""

        def _escape_sqlite_value(val: Any) -> Optional[str]:
            if isinstance(val, str):
                 # Escape single quotes for SQLite string literals
                 escaped_val = val.replace("'", "''")
                 return "'{}'".format(escaped_val)
            elif isinstance(val, bool):
                 return str(int(val)) # SQLite stores booleans as integers 0 or 1
            elif isinstance(val, (int, float)):
                return str(val)
            elif isinstance(val, (list, tuple)) and self.operator in [FilterOperator.IN, FilterOperator.NOT_IN]:
                formatted_elements = []
                for item in val:
                    if isinstance(item, (str, int, float, bool)):
                        formatted_elements.append(_escape_sqlite_value(item))
                    else:
                        return None
                return "({})".format(', '.join(e for e in formatted_elements if e is not None)) if all(e is not None for e in formatted_elements) else None
            elif val is None:
                 return 'NULL'
            return None

        # Convert field name from Python access (e.g., metadata["key"]) to SQLite JSON functions (e.g., json_extract(metadata, '$.key'))
        sqlite_field = self.field
        match = re.match(r"([a-zA-Z0-9_]+)\[[\"']?([a-zA-Z0-9_]+)[\"']?\]", self.field)
        if match:
            field_name, key_name = match.groups()
            # Check if the base field is 'metadata' (where JSON is stored)
            if field_name == 'metadata':
                 # Use json_extract for metadata access.
                 sqlite_field = f"json_extract({field_name}, '$.{key_name}')"
            else:
                 print(f"Warning: Field access syntax '{self.field}' used for non-'metadata' field in SQLite filter. Using raw field name.")
                 sqlite_field = self.field
        elif '[' in self.field or ']' in self.field:
             print(f"Warning: Complex field access '{self.field}' may not convert correctly to SQLite JSON functions. Using raw field name.")
             sqlite_field = self.field


        formatted_value = _escape_sqlite_value(self.value)
        if formatted_value is None:
            print(f"Warning: Cannot convert value {self.value} for field '{self.field}' to SQLite format.")
            return None

        sqlite_op = self.operator.value
        if self.operator == FilterOperator.LIKE:
             # SQLite LIKE uses single quotes and '%'/'_' wildcards.
             if isinstance(self.value, str):
                  # Escape single quotes for SQL string literal, then escape % and _ with \ for LIKE wildcards
                  escaped_val = self.value.replace("'", "''")
                  escaped_val = escaped_val.replace("%", "\\%").replace("_", "\\_") # Escape LIKE wildcards if they are literal
                  # Use the potentially converted sqlite_field (e.g., json_extract(...))
                  return f"{sqlite_field} LIKE '{escaped_val}' ESCAPE '\\'" # Specify escape character
             else:
                  print(f"Warning: LIKE operator requires string value for field '{self.field}'.")
                  return None
        elif self.operator in [FilterOperator.IN, FilterOperator.NOT_IN]:
             # SQLite IN syntax: field_name IN (value1, value2)
             if formatted_value.startswith('(') and formatted_value.endswith(')'):
                  # Use the potentially converted sqlite_field
                  return f"{sqlite_field} {sqlite_op} {formatted_value}"
             else:
                  print(f"Warning: Value for IN/NOT IN operator must be a list/tuple for field '{self.field}' for SQLite.")
                  return None
        elif self.operator == FilterOperator.CONTAINS:
             # Simulate CONTAINS for strings using LIKE
             if isinstance(self.value, str):
                  escaped_val = self.value.replace("'", "''").replace("%", "\\%").replace("_", "\\_")
                  # Use the potentially converted sqlite_field
                  return f"{sqlite_field} LIKE '%{escaped_val}%' ESCAPE '\\'"
             # CONTAINS for list elements within JSON field is complex and requires json_each or similar.
             print(f"Warning: CONTAINS operator for non-string values on field '{self.field}' in SQLite is not directly supported without JSON functions.")
             return None


        # Default comparison using the potentially converted field name
        return f"{sqlite_field} {sqlite_op} {formatted_value}"


class LogicalFilter(BaseFilter):
    """Represents a logical combination of filters (AND, OR, NOT)."""
    def __init__(self, operator: Union['AND', 'OR', 'NOT'], *filters: BaseFilter):
        if operator not in ['AND', 'OR', 'NOT']:
            raise ValueError("Logical operator must be 'AND', 'OR', or 'NOT'.")
        if operator in ['AND', 'OR'] and not filters:
            raise ValueError(f"{operator} filter must have at least one sub-filter.")
        if operator == 'NOT' and len(filters) != 1:
            raise ValueError("NOT filter must have exactly one sub-filter.")
        if operator == 'NOT' and not isinstance(filters[0], BaseFilter):
             raise ValueError("NOT filter must contain a valid filter object.")

        self.operator = operator
        self.filters = filters # Tuple of BaseFilter objects

    def to_milvus_expr(self) -> Optional[str]:
        """Converts LogicalFilter to Milvus/Milvus-lite expression string."""
        sub_exprs = [f.to_milvus_expr() for f in self.filters]
        valid_sub_exprs = [e for e in sub_exprs if e]

        if not valid_sub_exprs:
            return None

        if self.operator == 'AND':
            # Wrap sub-expressions in parentheses for safety
            return " and ".join(f"({e})" for e in valid_sub_exprs)
        elif self.operator == 'OR':
            # Wrap sub-expressions in parentheses for safety
            return " or ".join(f"({e})" for e in valid_sub_exprs)
        elif self.operator == 'NOT':
            return f"not ({valid_sub_exprs[0]})" # Only one sub-filter for NOT
        return None

    def to_chroma_filter(self) -> Optional[Dict[str, Any]]:
        """Converts LogicalFilter to ChromaDB metadata filter dictionary."""
        sub_filters = [f.to_chroma_filter() for f in self.filters]
        valid_sub_filters = [f for f in sub_filters if f]

        if not valid_sub_filters:
            return None

        if self.operator == 'AND':
            # Chroma uses '$and' with a list of filter dictionaries
            return {'$and': valid_sub_filters}
        elif self.operator == 'OR':
            # Chroma uses '$or' with a list of filter dictionaries
            return {'$or': valid_sub_filters}
        elif self.operator == 'NOT':
            # Chroma uses '$not' with a single filter dictionary
            return {'$not': valid_sub_filters[0]} # Only one sub-filter for NOT
        return None

    def to_qdrant_filter(self) -> Optional[models.Filter]:
        """Converts LogicalFilter to a Qdrant Filter object."""
        if not QDRANT_AVAILABLE: return None
        sub_qdrant_filters = [f.to_qdrant_filter() for f in self.filters]
        if self.operator == 'AND':
            all_must_conditions: List[models.Condition] = []
            # Assume AND requires all sub-filters to be true.
            valid_sub_filter_objects = [f for f in sub_qdrant_filters if f]
            if not valid_sub_filter_objects:
                 # print("Warning: AND logical filter has no valid sub-filters.")
                 return None
            return Filter(must=valid_sub_filter_objects)
        elif self.operator == 'OR':
             valid_sub_filter_objects = [f for f in sub_qdrant_filters if f]
             if not valid_sub_filter_objects:
                 # print("Warning: OR logical filter has no valid sub-filters.")
                 return None
             return Filter(should=valid_sub_filter_objects)

        elif self.operator == 'NOT':
            if len(self.filters) == 1:
                sub_filter = self.filters[0].to_qdrant_filter()
                if sub_filter:
                     return Filter(must_not=[sub_filter])
                else:
                     # print("Warning: NOT filter sub-filter conversion failed.")
                     return None
            else:
                print("Error: NOT filter requires exactly one sub-filter.")
                return None
        return None

    def to_sqlite_where(self) -> Optional[str]:
        """Converts LogicalFilter to SQLite WHERE clause string."""
        sub_wheres = [f.to_sqlite_where() for f in self.filters]
        valid_sub_wheres = [w for w in sub_wheres if w]

        if not valid_sub_wheres:
            return None

        if self.operator == 'AND':
            # Wrap sub-expressions in parentheses for safety
            return " AND ".join(f"({w})" for w in valid_sub_wheres)
        elif self.operator == 'OR':
            # Wrap sub-expressions in parentheses for safety
            return " OR ".join(f"({w})" for w in valid_sub_wheres)
        elif self.operator == 'NOT':
            # SQLite uses 'NOT (expression)' syntax
            return f"NOT ({valid_sub_wheres[0]})" # Only one sub-filter for NOT
        return None