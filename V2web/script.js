const hamburger = document.querySelector(".hamburger");
const navMenu = document.querySelector(".nav-menu");
const navBranding = document.querySelector(".nav-branding-2");

hamburger.addEventListener("click", () => {
  hamburger.classList.toggle("active");
  navMenu.classList.toggle("active");
  navBranding.classList.toggle("active");
});

document.querySelectorAll(".nav-link").forEach((n) =>
  n.addEventListener("click", () => {
    hamburger.classList.remove("active");
    navMenu.classList.remove("active");
  })
);

const quotes = [
    "Já jsem světlo světa; kdo mě následuje, nebude chodit ve tmě, ale bude mít světlo života.",
    "Milujte se navzájem, jako jsem já miloval vás.",
    "Já jsem cesta, pravda i život. Nikdo nepřichází k Otci než skrze mne.",
    "Neboť kde jsou dva nebo tři shromážděni ve jménu mém, tam jsem já uprostřed nich.",
    "Proste, a bude vám dáno; hledejte, a naleznete; tlučte, a bude vám otevřeno.",
    "Blahoslavení milosrdní, neboť oni dojdou milosrdenství.",
    "Blahoslavení čistého srdce, neboť oni uzří Boha.",
    "Nechte děti přicházet ke mně, nebraňte jim, neboť takovým patří království Boží.",
    "Pojďte ke mně všichni, kdo se namáháte a jste obtíženi břemeny, a já vám dám odpočinout.",
    "Hledejte především Boží království a jeho spravedlnost, a všechno ostatní vám bude přidáno.",
    "Království nebeské je jako poklad ukrytý v poli, který někdo najde a skryje; z radosti nad tím jde, prodá všecko, co má, a koupí to pole.",
    "Království nebeské je jako hořčičné zrno, které člověk zasel na svém poli; je sice menší než všecka semena, ale když vyroste, je větší než ostatní byliny.",
    "Království nebeské je jako kvas, který žena vmísí do tří měřic mouky, až se všecko prokvasí.",
    "Království Boží nepřichází tak, aby se to dalo pozorovat; ani se nedá říci: ‚Hle, je tu' nebo ‚je tam'! Vždyť království Boží je mezi vámi!",
    "Kdo se nepřijme království Boží jako dítě, jistě do něho nevejde.",
    "Jak těžko vejdou do království Božího ti, kdo mají bohatství! Snáze projde velbloud uchem jehly, než aby bohatý vešel do království Božího.",
    "Ne každý, kdo mi říká ‚Pane, Pane', vejde do království nebeského; ale ten, kdo činí vůli mého Otce v nebesích.",
    "Blaze chudým v duchu, neboť jejich je království nebeské.",
    "Hledejte nejprve Boží království a jeho spravedlnost, a všechno ostatní vám bude přidáno.",
    "Království nebeské je podobné králi, který vystrojil svatbu svému synu.",
    
    // Nové citáty
    "Dávejte a bude vám dáno; dobrá míra, natlačená, natřesená, vrchovatá vám bude dána do klína.",
    "Neboť jsem hladověl, a dali jste mi jíst, žíznil jsem, a dali jste mi pít, byl jsem na cestách, a ujali jste se mne.",
    "Kdo chce být první, buď ze všech poslední a služebník všech.",
    "Co prospěje člověku, získá-li celý svět, ale ztratí svou duši?",
    "Jako Otec miloval mne, tak já jsem miloval vás. Zůstaňte v mé lásce.",
    "Nové přikázání vám dávám, abyste se navzájem milovali; jako já jsem miloval vás.",
    "Já jsem vinný kmen, vy jste ratolesti. Kdo zůstává ve mně a já v něm, ten nese hojné ovoce.",
    "Váš Otec ví, co potřebujete, dříve než ho prosíte.",
    "Kde je tvůj poklad, tam bude i tvé srdce.",
    "Vy jste sůl země; jestliže však sůl pozbude chuti, čím bude osolena?",
    "Vy jste světlo světa. Nemůže zůstat skryto město ležící na hoře.",
    "Buďte dokonalí, jako je dokonalý váš nebeský Otec.",
    "Podle jejich ovoce je poznáte. Což sklízejí z trní hrozny nebo z bodláčí fíky?",
    "Každý strom, který nenese dobré ovoce, bude vyťat a hozen do ohně.",
    "Bděte tedy, protože nevíte, v který den váš Pán přijde.",

    // Nové citáty o lásce
    "Miluj Hospodina, Boha svého, celým svým srdcem, celou svou duší a celou svou myslí.",
    "Miluj svého bližního jako sám sebe.",
    "Větší lásku nemá nikdo než ten, kdo položí život za své přátele.",
    "Láska je trpělivá, láska je laskavá. Nezávidí, láska se nevychloubá a není domýšlivá.",
    "A tak zůstává víra, naděje a láska, ale největší z té trojice je láska.",
    "Kdo nemiluje, nepoznal Boha, protože Bůh je láska.",
    "V lásce není strach, ale dokonalá láska strach zahání.",
    "My milujeme, protože Bůh napřed miloval nás.",
    "Milujte své nepřátele a modlete se za ty, kdo vás pronásledují.",
    "Všechno dělejte v lásce.",

    // Nová podobenství
    "Podobno jest království nebeské zrnu hořčičnému, kteréž vzav člověk, vsál na poli svém. Kteréžto nejmenší jest mezi všemi semeny, ale když vzroste, větší jest než jiné byliny.",
    "Podobno jest království nebeské kvasu, kterýž vzavši žena, zadělala ve třech měřicích mouky, až by zkysalo všecko.",
    "Podobno jest království nebeské pokladu skrytému v poli, kterýž nalezna člověk, skryl, a radostí nad ním jde a prodá všecko, což má, a koupí pole to.",
    "Opět podobno jest království nebeské člověku kupci, hledajícímu pěkných perel. Kterýž když nalezl jednu velmi drahou perlu, odšel a prodal všecko, což měl, a koupil ji.",
    "Podobno jest království nebeské síti puštěné do moře a ze všelikého plodu rybího shromažďující.",
    "Podobno jest království nebeské člověku hospodáři, kterýž vyšel na úsvitě, aby najal dělníky na vinici svou.",
    "Podobno jest království nebeské člověku králi, kterýž učinil svatbu synu svému.",
    "Podobno bude království nebeské desíti pannám, kteréžto vzavše lampy své, vyšly proti ženichovi.",
    "Podobno jest království nebeské člověku, kterýž odcházeje z domu, povolal služebníků svých a dal jim statky své.",
    "Jako pastýř odděluje ovce od kozlů, tak budou odděleni spravedliví od nespravedlivých při příchodu Syna člověka.",

    // Citáty velkých myslitelů
    "Vím, že nic nevím. - Sokrates",
    "Člověk je měřítkem všech věcí. - Protagoras",
    "Poznej sám sebe. - Nápis v Delfách",
    "Celek je víc než souhrn jeho částí. - Aristoteles",
    "Nemůžeš dvakrát vstoupit do téže řeky. - Herakleitos",
    "Myslím, tedy jsem. - René Descartes",
    "Člověk je odsouzen ke svobodě. - Jean-Paul Sartre",
    "Pravda vás osvobodí. - Tomáš Akvinský",
    "Víra hledající porozumění. - Anselm z Canterbury",
    "Co není vědecké, není skutečné. - Auguste Comte",
    "Existence předchází esenci. - Jean-Paul Sartre",
    "Jednej tak, aby se maxima tvé vůle mohla stát principem všeobecného zákonodárství. - Immanuel Kant",
    "Život nezkoušený není hoden žití. - Sokrates",
    "Člověk je politický živočich. - Aristoteles",
    "Vědomí určuje bytí. - Karl Marx",
    "Kdo bojuje s nestvůrami, ať se má na pozoru, aby se sám nestal nestvůrou. - Friedrich Nietzsche",
    "Skepticismus je začátek víry. - Oscar Wilde",
    "Krása zachrání svět. - Fjodor Michajlovič Dostojevskij",
    "Láska k moudrosti je počátkem všeho poznání. - Platón",

    // Citáty Mahátmy Gándhího
    "Buď změnou, kterou chceš vidět ve světě. - Mahátma Gándhí",
    "Oko za oko učiní celý svět slepým. - Mahátma Gándhí",
    "Nejprve tě ignorují, pak se ti smějí, pak s tebou bojují, pak zvítězíš. - Mahátma Gándhí",
    "Život je jako zrcadlo, usmějete se na něj a on se usměje na vás. - Mahátma Gándhí",
    "Síla není v fyzické kapacitě, ale v neochvějné vůli. - Mahátma Gándhí",
    "Láska je nejsilnější silou, kterou svět disponuje. - Mahátma Gándhí",
    "Kde je láska, tam je život. - Mahátma Gándhí",
    "Pravda a nenásilí jsou staré jako hory. - Mahátma Gándhí",
    "Štěstí je harmonie mezi tím, co myslíte, říkáte a děláte. - Mahátma Gándhí",
    "Naše schopnost dosáhnout jednoty v různosti bude krásou civilizace. - Mahátma Gándhí",
    "Živý příklad má větší hodnotu než tisíc argumentů. - Mahátma Gándhí",
    "Svoboda není hodna toho jména, není-li svobodou mýlit se. - Mahátma Gándhí",
    "Nejlepší způsob, jak najít sebe sama, je ztratit se ve službě druhým. - Mahátma Gándhí",
    "Skromnost je pro morálku tím, čím je stín pro obraz. - Mahátma Gándhí",
    "Síla nenásilí je stokrát větší než síla zbraní. - Mahátma Gándhí",

    // Buddhovy citáty
    "Nenechte se vést pouze tím, co slyšíte. - Buddha",
    "V zdravém těle zdravý duch, toť nejkratší cesta ke štěstí. - Buddha",
    "Tisíc vítězství nad tisíci lidmi v bitvě se nevyrovná vítězství nad sebou samým. - Buddha",
    "Všechny jevy pocházejí z mysli. - Buddha",
    "Žij v přítomnosti, pamatuj na minulost a neboj se budoucnosti. - Buddha",
    "Nelpění je největší dar. - Buddha",
    "Láska a soucit jsou nutnosti, ne luxus. Bez nich lidstvo nemůže přežít. - Buddha",
    "Zdraví je největší dar, spokojenost největší bohatství, věrnost nejlepší vztah. - Buddha",
    "Slova mají sílu ničit i léčit. - Buddha",
    "Každé ráno se rodíme znovu. To, co uděláme dnes, je to, na čem záleží nejvíce. - Buddha",

    // Krišnovy citáty
    "Lepší je konat vlastní povinnost, byť nedokonale, než dokonale plnit povinnost druhého. - Krišna",
    "Člověk by měl pozdvihnout sám sebe a neměl by se ponižovat. - Krišna",
    "Jsem počátek, střed i konec všeho stvoření. - Krišna",
    "Moudrý člověk vidí utrpení v samotném požitku. - Krišna",
    "Mysl je přítel toho, kdo ji ovládl, ale nepřítelem toho, kdo ji neovládl. - Krišna",
    "Pracuj pro práci samotnou, ne pro její plody. - Krišna",
    "Ten, kdo vidí činnost v nečinnosti a nečinnost v činnosti, je moudrý mezi lidmi. - Krišna",
    "Štěstí pochází z klidu mysli. - Krišna",
    "Ovládni své smysly, ovládni svou duši. - Krišna",
    "V nevědomosti žijí ti, kdo vidí různost v jednotě. - Krišna"
];

// Vylepšení funkce pro zobrazování citátů s kategorií a autorem
function updateQuote() {
    const quoteElement = document.getElementById('quote');
    const randomIndex = Math.floor(Math.random() * quotes.length);
    const quote = quotes[randomIndex];
    
    quoteElement.style.opacity = 0;
    quoteElement.style.transform = 'translateY(20px)';
    
    setTimeout(() => {
        quoteElement.textContent = quote;
        quoteElement.style.opacity = 1;
        quoteElement.style.transform = 'translateY(0)';

        // Rozpoznání typu citátu a aplikace stylu
        if (quote.includes("- Mahátma Gándhí")) {
            quoteElement.className = 'gandhi-quote';
        } else if (quote.includes("- Buddha")) {
            quoteElement.className = 'buddha-quote';
        } else if (quote.startsWith("Podobno jest království") || quote.includes("království") || quote.includes("Království")) {
            quoteElement.className = 'kingdom-quote';
        } else if (!quote.includes("-")) {
            quoteElement.className = 'jesus-quote';
        } else {
            quoteElement.className = 'philosopher-quote';
        }
    }, 500);
}

// Aktualizace citátu každých 10 sekund
document.addEventListener('DOMContentLoaded', () => {
    updateQuote();
    setInterval(updateQuote, 15000);
});

(function adjustBodyPadding(){
        const header = document.getElementById('header');
        if (!header) return;
        function update() {
          const h = header.offsetHeight;
          document.body.style.paddingTop = h + 'px';
        }
        window.addEventListener('resize', update);
        document.addEventListener('DOMContentLoaded', update);
        // okamžité nastavení pokud skript běží po načtení
        update();
      })();