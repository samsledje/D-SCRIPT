'use strict';

const Code = require('code');
const Lab = require('lab');
const Reach = require('../lib');

// Test shortcuts
const lab = exports.lab = Lab.script();
const expect = Code.expect;
const describe = lab.describe;
const it = lab.it;


describe('Reach', () => {
  const obj = {
    a: {
      b: {
        c: {
          d: 1,
          e: 2
        },
        f: 'hello'
      },
      g: {
        h: 3
      }
    },
    i: function () { },
    j: null,
    k: [4, 8, 9, 1]
  };

  obj.i.x = 5;

  it('returns first value of array', (done) => {
    expect(Reach(obj, 'k.0')).to.equal(4);
    done();
  });

  it('returns last value of array using negative index', (done) => {
    expect(Reach(obj, 'k.-2')).to.equal(9);
    done();
  });

  it('returns a valid member', (done) => {
    expect(Reach(obj, 'a.b.c.d')).to.equal(1);
    done();
  });

  it('returns a valid member with separator override', (done) => {
    expect(Reach(obj, 'a/b/c/d', { separator: '/' })).to.equal(1);
    done();
  });

  it('returns undefined on null object', (done) => {
    expect(Reach(null, 'a.b.c.d')).to.equal(undefined);
    done();
  });

  it('returns undefined on missing object member', (done) => {
    expect(Reach(obj, 'a.b.c.d.x')).to.equal(undefined);
    done();
  });

  it('returns undefined on missing function member', (done) => {
    expect(Reach(obj, 'i.y')).to.equal(undefined);
    done();
  });

  it('throws on missing member in strict mode', (done) => {
    expect(() => {
      Reach(obj, 'a.b.c.o.x', { strict: true });
    }).to.throw('Invalid segment, o, in reach path a.b.c.o.x.');

    done();
  });

  it('returns undefined on invalid member', (done) => {
    expect(Reach(obj, 'a.b.c.d-.x')).to.equal(undefined);
    done();
  });

  it('returns function member', (done) => {
    expect(typeof Reach(obj, 'i')).to.equal('function');
    done();
  });

  it('returns function property', (done) => {
    expect(Reach(obj, 'i.x')).to.equal(5);
    done();
  });

  it('returns null', (done) => {
    expect(Reach(obj, 'j')).to.equal(null);
    done();
  });

  it('will return a default value if property is not found', (done) => {
    expect(Reach(obj, 'a.b.q', { default: 'foo' })).to.equal('foo');
    done();
  });

  it('will return a default value if path is not found', (done) => {
    expect(Reach(obj, 'q', { default: 'foo' })).to.equal('foo');
    done();
  });

  it('allows a falsey value to be used as the default value', (done) => {
    expect(Reach(obj, 'q', { default: '' })).to.equal('');
    done();
  });

  it('throws if chain is not a string', (done) => {
    function fail (obj, chain) {
      expect(() => {
        Reach(obj, chain);
      }).to.throw(`Reach path must a string. Found ${chain}.`);
    }

    fail(obj);
    fail(obj, null);
    fail(obj, []);
    fail(obj, {});
    fail(obj, true);
    fail(obj, false);
    done();
  });
});
