# reach

[![Current Version](https://img.shields.io/npm/v/reach.svg)](https://www.npmjs.org/package/reach)
[![Build Status via Travis CI](https://travis-ci.org/cjihrig/reach.svg?branch=master)](https://travis-ci.org/cjihrig/reach)
![Dependencies](http://img.shields.io/david/cjihrig/reach.svg)
[![belly-button-style](https://cdn.rawgit.com/cjihrig/belly-button/master/badge.svg)](https://github.com/cjihrig/belly-button)

Safely retrieve nested object keys. Inspired by the [Hoek](https://github.com/hapijs/hoek) module's `reach()` method.

```javascript
const Reach = require('reach');
const obj = {
  foo: {
    bar: {
      baz: 3
    }
  }
};

Reach(obj, 'foo.bar.baz');
// Returns 3
```

## Methods

`Reach` exports a single function, described below.

### `reach(obj, chain [, options])`

  - Arguments
    - `obj` (object) - An object to retrieve a value from.
    - `chain` (string) - A string specifying the path to traverse within `obj`. Path segments are delimited by periods (`'.'`) by default. If a non-string is provided, a `TypeError` is thrown.
    - `options` (object) - A configuration object supporting the following keys.
      - `separator` (string) - Path segment delimiter. Defaults to `'.'`.
      - `strict` (boolean) - If `true`, an error is thrown when the complete `chain` cannot be found in `obj`. Defaults to `false`.
      - `default` - The value returned if the complete `chain` cannot be found in `obj`, and `strict` is `false`. Defaults to `undefined`.
  - Returns
    - The value found by traversing `chain` through `obj`. If no value is found, and the `strict` option is `false` (default behavior), then `default` is returned.

Traverses an object, `obj`. The path through the object is dictated by the `chain` string.
