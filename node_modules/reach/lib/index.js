'use strict';

module.exports = reach;


const defaults = {
  separator: '.',
  strict: false,
  default: undefined
};


function reach (obj, chain, options) {
  if (typeof chain !== 'string') {
    throw new TypeError(`Reach path must a string. Found ${chain}.`);
  }

  const settings = Object.assign({}, defaults, options);
  const path = chain.split(settings.separator);
  let ref = obj;

  for (let i = 0; i < path.length; ++i) {
    let key = path[i];

    if (key[0] === '-' && Array.isArray(ref)) {
      key = key.slice(1, key.length);
      key = ref.length - key;
    }

    // ref must be an object or function and contain key
    if (ref === null ||
        (typeof ref !== 'object' && typeof ref !== 'function') ||
        !(key in ref)) {
      if (settings.strict) {
        throw new Error(`Invalid segment, ${key}, in reach path ${chain}.`);
      }

      return settings.default;
    }

    ref = ref[key];
  }

  return ref;
}
