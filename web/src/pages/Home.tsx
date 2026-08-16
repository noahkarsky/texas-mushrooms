import { Link } from 'react-router-dom'
import { HERO_DOTS } from '../heroDots'

const TEXAS_MUSHROOMS_URL = 'https://www.texasmushrooms.org/'
const REPO_URL = 'https://github.com/noahkarsky/texas-mushrooms'
const COLOR_DOCS_URL =
  'https://github.com/noahkarsky/texas-mushrooms/blob/main/docs/color-measurement.md'
const INAT_URL = 'https://www.inaturalist.org/'

/**
 * Deterministic vertical scatter. Must not be `Math.random()`: the dots have to
 * land in the same place across re-renders and between dev and prod, or the
 * field visibly reshuffles on every state change.
 */
function jitter(i: number): number {
  return ((i * 2654435761) % 1000) / 1000
}

/**
 * A miniature of the Seasons chart rather than decoration: x is day-of-year and
 * the fill is that photograph's own measured color, so the hero teaches the
 * encoding before the visitor ever reaches `/seasons`.
 *
 * Absolutely-positioned elements, not SVG circles — an SVG stretched to the
 * container width with `preserveAspectRatio="none"` would squash the circles
 * into ellipses. Percentage offsets keep them round at any width with no
 * resize observer.
 */
function HeroField() {
  return (
    <div className="home-hero" aria-hidden="true">
      {HERO_DOTS.map((entry, i) => {
        const [doy, hex] = entry.split(':')
        const day = Number(doy)
        return (
          <span
            key={i}
            className="home-dot"
            style={{
              left: `${(day / 366) * 100}%`,
              top: `${6 + jitter(i) * 88}%`,
              background: `#${hex}`,
              animationDelay: `${(day / 366) * 600}ms`,
            }}
          />
        )
      })}
    </div>
  )
}

function Stat({ n, label }: { n: string; label: string }) {
  return (
    <div>
      <div className="home-stat-n">{n}</div>
      <div className="home-stat-l">{label}</div>
    </div>
  )
}

function Finding({
  title,
  coef,
  interval,
  tone = 'wet',
  children,
}: {
  title: string
  coef: string
  interval: string
  tone?: 'wet' | 'dry' | 'null'
  children: React.ReactNode
}) {
  return (
    <div className="home-finding">
      <div className="home-finding-num">
        <div className={`home-coef is-${tone}`}>{coef}</div>
        <div className="home-interval">
          94% HDI
          <br />
          {interval}
        </div>
      </div>
      <div>
        <h3>{title}</h3>
        <p>{children}</p>
      </div>
    </div>
  )
}

function Out({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a className="home-a" href={href} target="_blank" rel="noopener noreferrer">
      {children}
    </a>
  )
}

export default function HomePage() {
  return (
    <main className="home">
      <div className="home-inner">
        <h1 className="home-h1">Texas Mushrooms</h1>
        <p className="home-dek">
          Seventeen years of one man&rsquo;s mushroom photographs, turned into a map and a calendar.
        </p>

        <HeroField />
        <p className="home-caption">
          Each dot is one of 400 photographs, placed by the day of the year it was taken and painted
          in the mushroom&rsquo;s own measured color. The full picture &mdash; all 8,800 &mdash; is on
          the Seasons page.
        </p>

        <div className="home-stats">
          <Stat n="8,800" label="photographs mapped" />
          <Stat n="718" label="species identified" />
          <Stat n="2018–2024" label="seasons covered" />
        </div>

        <div className="home-cta">
          <Link to="/map">Open the map</Link>
          <Link to="/seasons">See the seasons</Link>
        </div>

        <section className="home-section">
          <h2 className="home-h2">One person, one camera, seventeen years</h2>
          <p className="home-p">
            Alexey Sergeev teaches mathematics at Texas A&amp;M in College Station, and has been
            publishing his mushroom walks at{' '}
            <Out href={TEXAS_MUSHROOMS_URL}>texasmushrooms.org</Out> since October 2007 &mdash;
            19,156 pictures of 1,232 species by his own count, and still growing.
          </p>
          <p className="home-p">
            He carries a GPS. Every day page ships a track log whose waypoint names encode the same
            roll-and-frame keys as the photo filenames, which is why 98.5% of these photographs
            carry real coordinates instead of a vague locality string. His captions are
            natural-history prose: they name the place, the substrate, and often the color outright
            &mdash; which is the only reason the color extraction below could be checked against
            anything.
          </p>
          <p className="home-p">
            This site is built from that public archive. It is not affiliated with him.
          </p>
        </section>

        <section className="home-section">
          <h2 className="home-h2">The map: where one person&rsquo;s route actually goes</h2>
          <p className="home-p">
            His 8,800 photographs fall inside 46 hexagonal cells. iNaturalist, over the same
            bounding box, covers 1,342 cells and 16,137 photographs. That gap is not a
            data-quality difference &mdash; it is the difference between a crowd and a route.
          </p>
          <p className="home-p">
            6,916 of his photographs (79%) come from Sam Houston National Forest; the rest from Big
            Creek Scenic Area, Big Thicket National Preserve, and Huntsville State Park. The map
            colors each cell by photo count or mean elevation, and toggles between the two sources
            so you can see the contrast directly.{' '}
            <Link className="home-a" to="/map">
              Open the map
            </Link>
            .
          </p>
        </section>

        <section className="home-section">
          <h2 className="home-h2">The seasons: 8,800 photographs, by the day they were taken</h2>
          <p className="home-p">
            Every dot is a single photograph, placed by the day it was taken and painted in the
            mushroom&rsquo;s own colors. The background ribbon runs blue in wetter-than-normal
            stretches and amber in drier ones, one row per year, 2018 through 2024.
          </p>
          <p className="home-p">
            The color is measured from the photograph &mdash; a weighted k-means in a
            perceptual color space over the pixels least like the frame&rsquo;s background &mdash;
            not assigned from a species lookup. A second view stacks the genera by peak fruiting
            day, earliest at the top.{' '}
            <Link className="home-a" to="/seasons">
              See the seasons
            </Link>
            .
          </p>
        </section>

        <section className="home-section">
          <h2 className="home-h2">What the numbers say</h2>
          <p className="home-p">
            Two zero-inflated Poisson models, fit on 2,511 days of Open-Meteo weather and on the
            cells themselves.
          </p>
          <div className="home-findings">
            <Finding title="Soil moisture drives fruiting" coef="+0.434" interval="[0.372, 0.503]">
              About 1.54&times; the species count per standard deviation of mean soil moisture. The
              strongest effect in the model, and the interval is nowhere near zero.
            </Finding>
            <Finding
              title="Last week's rain does not"
              coef="−0.029"
              interval="[−0.073, 0.014]"
              tone="null"
            >
              Seven-day rainfall totals show <strong>no detectable effect</strong> &mdash; the
              interval spans zero. Cumulative rain is the folk-wisdom predictor; what the model says
              is that what is already in the ground beats what fell from the sky last week.
            </Finding>
            <Finding
              title="Lower ground fruits more"
              coef="−0.298"
              interval="[−0.317, −0.280]"
              tone="dry"
            >
              Roughly 26% fewer photographs per standard deviation higher in elevation, across a
              26&ndash;135 m range. In this landscape, lower is wetter. Seasonality is real and
              strong in both models.
            </Finding>
          </div>
        </section>

        <section className="home-section">
          <h2 className="home-h2">What this is not</h2>
          <p className="home-p">
            The spatial model describes where Sergeev walks, not where Texas fungi grow. 36% of the
            photographs come from a single cell, and the elevation result inherits that bias
            wholesale. Read it as a model of an archive, not of a landscape.
          </p>
          <p className="home-p">
            The color extraction scores 0.232 accuracy under five-fold cross-validation against
            roughly 1,890 photographer-stated caption colors, against a chance floor near 0.04.
            Better than chance and better than the whole-frame extractor it replaced, but it is a
            measurement with known regressions, not a classifier &mdash; the{' '}
            <Out href={COLOR_DOCS_URL}>measurement notes</Out> document where it fails.
          </p>
          <p className="home-p">
            No photographs are served here. The source site uses hotlink protection, and a public
            mirror should not push its traffic onto someone else&rsquo;s server. What you get is the
            colors, the coordinates, the dates, and a link back to the original page.
          </p>
        </section>

        <footer className="home-footer">
          <div className="home-footer-links">
            <Out href={TEXAS_MUSHROOMS_URL}>texasmushrooms.org</Out>
            <Out href={REPO_URL}>Source on GitHub</Out>
            <Out href={INAT_URL}>iNaturalist</Out>
          </div>
          Photographs and captions are Alexey Sergeev&rsquo;s. The scraping, modeling, and these
          visualizations are not affiliated with him.
        </footer>
      </div>
    </main>
  )
}
