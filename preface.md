# About the Scientific Python Lectures

```{contents}
:depth: 1
:local: true
```

% Hack to have multi-column layout in authors list

*Release:* {{ release }}

![http://dx.doi.org/10.5281/zenodo.594102](https://zenodo.org/badge/doi/10.5281/zenodo.594102.svg)

```{raw} html
<style type="text/css">
  div.section#authors ul.simple li,
  div.section#requirements ul.simple li{
      float: left;
      min-width: 200px;
      width: 30%;
      margin-right: 1.25em;
  }

  /* Below is necessary to avoid messing up android font size */
  @media  only screen and (max-width: 1080px) and (-webkit-min-device-pixel-ratio: 2.5),
      only screen and (max-width: 1080px) and (-o-min-device-pixel-ratio: 25/10),
      only screen and (max-width: 1080px) and (min-resolution: 250dpi)
  {
      div.section#authors ul.simple li,
      div.section#requirements ul.simple li{
          float: none;
          width: 100%;
          min-width: 100%;
          margin-right: 0pt;
      }
  }

  div.section#authors ul.simple:after,
  div.section#requirements ul.simple li:after {
      display: block;
      font-size: 0;
      content: " ";
      clear: both;
      height: 1em;
  }
</style>
```

```{eval-rst}
.. include:: AUTHORS.rst
```

```{eval-rst}
.. include:: CHANGES.rst
```

```{eval-rst}
.. include:: LICENSE.rst
```

```{eval-rst}
.. include:: CONTRIBUTING.rst
```
